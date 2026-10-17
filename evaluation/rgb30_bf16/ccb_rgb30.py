#!/usr/bin/env python3
"""Portable RGB30 runner. No credentials, human responses, or default remote endpoints."""
from pathlib import Path
import argparse,base64,csv,datetime,fcntl,hashlib,importlib.metadata,io,json,os,re,subprocess,sys,time,uuid,zipfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from response_v12 import parse_response,ResponseValidationError
from container_repair_source import repair_json_container
import metrics_v1 as metrics
import holistic_alignment_baseline as baseline
import httpx
from PIL import Image

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def config(name):
 path=ROOT/'configs'/name
 if path.suffix!='.json':path=path.with_suffix('.json')
 assert path.parent==ROOT/'configs' and path.exists(),'Unknown config'
 return read(path)
def verify():
 m=read(ROOT/'frozen_input_manifest.json');plan=read(ROOT/'run_plan.json')
 assert sha(ROOT/'frozen_input_manifest.json')==plan['input_manifest_sha256']
 assert len(m['formal_task_ids'])==30 and len(set(m['formal_task_ids']))==30
 original=read(ROOT/'provenance/original_frozen_input_manifest.json')
 assert sha(ROOT/'provenance/original_frozen_input_manifest.json')==m['source_manifest_sha256']
 assert m['tasks']==original['tasks'] and m['formal_task_ids']==original['formal_task_ids']
 for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
 for t in m['tasks']:
  for prefix in ['image','prompt','task']:assert sha(ROOT/t[prefix+'_path'])==t[prefix+'_sha256']
  assert t['canonical_mapping']=={x:x for x in 'ABCD'}
 lock=ROOT/'bundle_manifest.json'
 if lock.exists():
  for name,digest in read(lock)['files'].items():assert sha(ROOT/name)==digest,'Bundle changed: '+name
 return m,plan

def parse(raw):
 schema=read(ROOT/'protocol/response.schema.json')
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ResponseValidationError('duplicate_json_key')
   out[k]=v
  return out
 for candidate,method in [(raw,None),repair_json_container(raw)]:
  try:
   json.loads(candidate,object_pairs_hook=unique)
   parsed=parse_response(candidate,schema)
   totals={o:sum(parsed['ratings'][o].values()) for o in 'ABCD'}
   derived=[[o for o in 'ABCD' if totals[o]==v] for v in sorted(set(totals.values()),reverse=True)]
   return parsed,method,totals,derived
  except (ValueError,TypeError):pass
 raise ResponseValidationError('No valid explicit holistic response')

def body(cfg,prompt,image_bytes,served_id):
 payload=dict(model=served_id,messages=[dict(role='user',content=[dict(type='text',text=prompt),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode()))])],**cfg['generation'])
 if cfg['enable_thinking']:payload['chat_template_kwargs']={'enable_thinking':True}
 return payload

def classify(provider):
 choices=provider.get('choices') or []
 if not choices:return dict(status='missing_choices',valid=False)
 choice=choices[0];msg=choice.get('message') or {};raw=msg.get('content')
 if msg.get('refusal'):return dict(status='refusal',valid=False)
 if not isinstance(raw,str) or not raw.strip():return dict(status='truncated' if choice.get('finish_reason')=='length' else 'empty_response',valid=False)
 try:
  parsed,repair,totals,derived=parse(raw)
  return dict(status='valid',valid=True,parsed=parsed,repair_method=repair,dimension_totals=totals,dimension_ranking=derived)
 except ResponseValidationError:
  candidate,_=repair_json_container(raw)
  try:
   obj=json.loads(candidate);status='missing_fields' if isinstance(obj,dict) and not {'ratings','overall_ranking','confidence'}<=set(obj) else 'schema_failure'
  except (ValueError,TypeError):status='parse_failure'
  if choice.get('finish_reason')=='length':status='truncated'
  return dict(status=status,valid=False)

def endpoint(state):return 'http://127.0.0.1:'+str(int(state['port']))+'/v1'
def state_for(cfg):
 state=read(ROOT/'deployment'/cfg['key']/'state.json')
 assert state['config']==cfg and state['config_sha256']==sha(ROOT/'configs'/(cfg['key']+'.json'))
 os.kill(state['server_pid'],0)
 # Check that the live endpoint advertises this launch's unique model name.
 with httpx.Client(timeout=15,trust_env=False) as client:
  r=client.get(endpoint(state)+'/models');r.raise_for_status()
  assert state['served_model_id'] in [x['id'] for x in r.json()['data']],'Wrong server'
 return state

def request(cfg,state,payload,dest):
 dest.mkdir(parents=True,exist_ok=True)
 # Store exact text/settings and image hash; image bytes are already frozen in inputs/.
 stored=json.loads(json.dumps(payload));url=stored['messages'][0]['content'][1]['image_url'].pop('url')
 stored['messages'][0]['content'][1]['image_url']['frozen_image_sha256']=hashlib.sha256(base64.b64decode(url.split(',',1)[1])).hexdigest()
 write(dest/'request.json',stored)
 started=time.monotonic()
 try:
  with httpx.Client(timeout=cfg['timeout_seconds'],trust_env=False) as client:
   r=client.post(endpoint(state)+'/chat/completions',json=payload)
  # Save raw HTTP body before parsing, including malformed provider JSON.
  (dest/'provider_response.txt').write_bytes(r.content)
  if r.status_code>=400:return dict(status='http_'+str(r.status_code),valid=False,latency_seconds=time.monotonic()-started)
  try:provider=r.json()
  except ValueError:return dict(status='provider_json_failure',valid=False)
  write(dest/'provider_response.json',provider)
  msg=(provider.get('choices') or [{}])[0].get('message') or {}
  if isinstance(msg.get('content'),str):(dest/'raw_response.txt').write_text(msg['content'])
  if isinstance(msg.get('reasoning_content') or msg.get('reasoning'),str):(dest/'reasoning.txt').write_text(msg.get('reasoning_content') or msg.get('reasoning'))
  record=classify(provider)
  record.update(returned_model_id=provider.get('model'),usage=provider.get('usage'),finish_reason=(provider.get('choices') or [{}])[0].get('finish_reason'),latency_seconds=time.monotonic()-started)
  if provider.get('model')!=state['served_model_id']:record.update(status='model_id_mismatch',valid=False)
  if 'parsed' in record:write(dest/'parsed.json',record['parsed'])
  return record
 except httpx.TimeoutException:return dict(status='timeout',valid=False,latency_seconds=time.monotonic()-started)
 except httpx.TransportError:return dict(status='connection',valid=False,latency_seconds=time.monotonic()-started)

def retryable(status):return status in ['timeout','connection','http_429','interrupted_unknown'] or bool(re.fullmatch('http_5[0-9][0-9]',status))

def serve(args):
 verify();cfg=config(args.model)
 assert importlib.metadata.version('vllm')==cfg['backend_version'],'Install pinned server environment'
 gpu_ids=args.gpus.split(',');assert len(gpu_ids)==cfg['tensor_parallel_size'] and len(set(gpu_ids))==len(gpu_ids)
 assert all(re.fullmatch(r'\d+',x) for x in gpu_ids)
 from huggingface_hub import snapshot_download
 directory=ROOT/'deployment'/cfg['key'];directory.mkdir(parents=True,exist_ok=True)
 with (directory/'serve.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Downloads original weights only. No GGUF, AWQ, GPTQ, FP8, CPU offload or model substitution.
  snapshot=Path(snapshot_download(cfg['repo_id'],revision=cfg['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','LICENSE*'],token=False))
  assert snapshot.name==cfg['revision']
  model_config=read(snapshot/'config.json')
  assert not model_config.get('quantization_config'),'Quantized checkpoint forbidden'
  weight_files=sorted(snapshot.glob('*.safetensors'));assert weight_files,'No safetensors weights'
  inventory={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in sorted(snapshot.iterdir()) if p.is_file()}
  served=cfg['key']+'__'+cfg['revision'][:12]+'__'+uuid.uuid4().hex[:8]
  command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(snapshot),'--served-model-name',served,'--dtype','bfloat16','--cpu-offload-gb','0','--kv-cache-dtype','auto','--host','127.0.0.1','--port',str(args.port),'--tensor-parallel-size',str(cfg['tensor_parallel_size']),'--max-model-len',str(cfg['max_model_len']),'--max-num-seqs','1','--gpu-memory-utilization',str(cfg['gpu_memory_utilization']),'--limit-mm-per-prompt','{"image":1}','--generation-config','vllm','--seed','42']
  if cfg['reasoning_parser']:command+=['--reasoning-parser',cfg['reasoning_parser']]
  session_dir=directory/served;session_dir.mkdir()
  (session_dir/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
  (session_dir/'nvidia_smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus)
  with (session_dir/'server.log').open('w') as log:
   process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
   state=dict(config=cfg,config_sha256=sha(ROOT/'configs'/(cfg['key']+'.json')),served_model_id=served,port=args.port,server_pid=process.pid,started_at=now(),gpu_ids=gpu_ids,command=command,model_revision=cfg['revision'],snapshot_inventory=inventory,session_dir=relative(session_dir),deployment_validation='awaiting_synthetic_smoke')
   write(directory/'state.json',state);write(session_dir/'state.json',state)
   print('Server starting; inspect '+str(session_dir/'server.log'),flush=True)
   try:code=process.wait()
   except KeyboardInterrupt:
    process.terminate();code=process.wait()
   write(session_dir/'exit.json',dict(exit_code=code,ended_at=now()))
   if code:raise SystemExit('Server failed; preserve log and mark BLOCKED. Do not change frozen config in-place.')

def smoke(args):
 verify();cfg=config(args.model);state=state_for(cfg)
 dest=ROOT/state['session_dir']/'smoke'
 assert not dest.exists(),'Smoke already recorded; inspect it, do not retry silently'
 buffer=io.BytesIO();Image.new('RGB',(1449,1087),(36,94,160)).save(buffer,format='JPEG')
 payload=body(cfg,'What is the dominant color of this image? Answer briefly.',buffer.getvalue(),state['served_model_id'])
 result=request(cfg,state,payload,dest)
 provider=read(dest/'provider_response.json') if (dest/'provider_response.json').exists() else {}
 message=(provider.get('choices') or [{}])[0].get('message') or {}
 ok=provider.get('model')==state['served_model_id'] and isinstance(message.get('content'),str) and bool(message['content'].strip()) and not message.get('refusal') and (provider.get('choices') or [{}])[0].get('finish_reason')!='length'
 if cfg['enable_thinking']:ok=ok and bool(message.get('reasoning_content') or message.get('reasoning'))
 write(dest/'smoke_result.json',dict(passed=ok,kind='synthetic_non_benchmark_not_a_formal_response',completed_at=now(),model_id=provider.get('model'),thinking_output_present=bool(message.get('reasoning_content') or message.get('reasoning')),request_status=result['status']))
 assert ok,'BLOCKED: synthetic RGB/reasoning transport smoke failed. Inspect logs, no formal calls made.'
 print('Synthetic RGB and reasoning transport check passed; not a model accuracy test.')

def run(args):
 manifest,plan=verify();cfg=config(args.model);state=state_for(cfg)
 assert read(ROOT/state['session_dir']/'smoke/smoke_result.json')['passed'],'Run smoke first'
 directory=ROOT/'runs'/cfg['key'];directory.mkdir(parents=True,exist_ok=True)
 with (directory/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  identity=dict(config=cfg,input_manifest_sha256=sha(ROOT/'frozen_input_manifest.json'),source_manifest_sha256=manifest['source_manifest_sha256'],run_plan_sha256=sha(ROOT/'run_plan.json'),bundle_manifest_sha256=sha(ROOT/'bundle_manifest.json'),formal_task_ids=manifest['formal_task_ids'])
  path=directory/'run_manifest.json'
  if path.exists():
   run_manifest=read(path);assert run_manifest['identity']==identity,'Resume identity mismatch'
  else:
   run_manifest=dict(identity=identity,started_at=now(),run_id=cfg['key']+'__'+uuid.uuid4().hex,deployments=[],status='running');write(path,run_manifest)
  if state['session_dir'] not in run_manifest['deployments']:run_manifest['deployments'].append(state['session_dir'])
  write(path,run_manifest)
  for task in manifest['tasks']:
   taskdir=directory/'tasks'/task['task_id'];final=taskdir/'result.json'
   if final.exists():continue
   for number in [1,2]:
    next_dest=taskdir/('attempt-%02d'%number)
    if number==2 and not next_dest.exists():
     used=sum(1 for _ in directory.glob('tasks/*/attempt-02/attempt.json'))
     if used>=plan['retry_policy']['max_extra_attempts_per_model']:break
     time.sleep(plan['retry_policy']['delay_seconds'])
    dest=next_dest
    if (dest/'outcome.json').exists():record=read(dest/'outcome.json')
    elif (dest/'attempt.json').exists():
     # Recover a saved response locally. Never replay a saved malformed/refused response.
     if (dest/'provider_response.json').exists():
      provider=read(dest/'provider_response.json');record=classify(provider)
      message=(provider.get('choices') or [{}])[0].get('message') or {}
      if isinstance(message.get('content'),str):(dest/'raw_response.txt').write_text(message['content'])
      record['returned_model_id']=provider.get('model')
      if provider.get('model')!=read(dest/'attempt.json')['served_model_id']:record.update(valid=False,status='model_id_mismatch')
     elif (dest/'provider_response.txt').exists():record=dict(status='saved_unparsed_http_response',valid=False)
     else:record=dict(status='interrupted_unknown',valid=False)
     write(dest/'outcome.json',record)
    else:
     dest.mkdir(parents=True,exist_ok=True)
     write(dest/'attempt.json',dict(started_at=now(),task_id=task['task_id'],attempt=number,served_model_id=state['served_model_id'],input_manifest_sha256=identity['input_manifest_sha256'],image_sha256=task['image_sha256'],prompt_sha256=task['prompt_sha256'],deployment=state['session_dir']))
     record=request(cfg,state,body(cfg,(ROOT/task['prompt_path']).read_text(),(ROOT/task['image_path']).read_bytes(),state['served_model_id']),dest)
     record['completed_at']=now();write(dest/'outcome.json',record)
    if not retryable(record['status']) or number==2:break
   if (dest/'raw_response.txt').exists():record['raw_response_sha256']=sha(dest/'raw_response.txt')
   record.update(task_id=task['task_id'],selected_attempt_path=relative(dest),input_manifest_sha256=identity['input_manifest_sha256'],image_sha256=task['image_sha256'],prompt_sha256=task['prompt_sha256'])
   write(final,record);print(task['task_id'],record['status'],flush=True)
   if record['status'] in ['http_401','http_403','http_404','model_id_mismatch']:
    run_manifest['status']='BLOCKED';write(path,run_manifest);raise SystemExit('Endpoint/version blocked; remaining slots retained as not_run')
  results=[read(p) for p in directory.glob('tasks/*/result.json')]
  run_manifest.update(completed_at=now(),actual_attempts=sum(1 for _ in directory.glob('tasks/*/attempt-*/attempt.json')),valid_responses=sum(bool(r.get('valid')) for r in results),completed_slots=len(results),status='completed' if len(results)==30 else 'partial')
  write(path,run_manifest)
  export(args)

def export(args):
 manifest,_=verify();cfg=config(args.model);directory=ROOT/'runs'/cfg['key']
 identity=read(directory/'run_manifest.json')['identity']
 assert identity['config']==cfg and identity['input_manifest_sha256']==sha(ROOT/'frozen_input_manifest.json')
 assert identity['run_plan_sha256']==sha(ROOT/'run_plan.json') and identity['bundle_manifest_sha256']==sha(ROOT/'bundle_manifest.json')
 coverage=[];dims=[];partial=[];failures=[];predictions=[]
 for task in manifest['tasks']:
  tid=task['task_id'];p=directory/'tasks'/tid/'result.json';r=read(p) if p.exists() else dict(status='not_run',valid=False)
  coverage.append(dict(model_key=cfg['key'],task_id=tid,valid=bool(r.get('valid')),status=r['status'],attempts=sum(1 for _ in (directory/'tasks'/tid).glob('attempt-*/attempt.json'))))
  if p.exists():
   assert r['input_manifest_sha256']==identity['input_manifest_sha256']
   assert r['image_sha256']==task['image_sha256'] and r['prompt_sha256']==task['prompt_sha256']
   selected=(ROOT/r['selected_attempt_path']).resolve()
   assert selected.parent==(directory/'tasks'/tid).resolve() and selected.name in ['attempt-01','attempt-02']
   if r.get('raw_response_sha256'):assert sha(selected/'raw_response.txt')==r['raw_response_sha256']
  if r.get('valid'):
   parsed,repair,totals,derived=parse((ROOT/r['selected_attempt_path']/'raw_response.txt').read_text());assert parsed==r['parsed']
   predictions.append(dict(task_id=tid,overall_ranking=parsed['overall_ranking']))
   for o in 'ABCD':dims.append(dict(model_key=cfg['key'],task_id=tid,option=o,**parsed['ratings'][o],E_F_C_total=totals[o],derived_ranking=derived,direct_holistic=parsed['overall_ranking']))
  for outcome in (directory/'tasks'/tid).glob('attempt-*/outcome.json'):
   rec=read(outcome);failures.append(dict(task_id=tid,attempt=outcome.parent.name,status=rec['status'],repair_method=rec.get('repair_method')))
   raw=outcome.parent/'raw_response.txt'
   if not rec.get('valid') and raw.exists():
    saved=metrics.partial_ratings(raw.read_text(),read(ROOT/'protocol/response.schema.json'))
    if saved:
     ratings,totals,derived=saved
     for o in 'ABCD':partial.append(dict(task_id=tid,option=o,**ratings[o],E_F_C_total=totals[o],derived_ranking=derived,holistic_eligible=False))
 for name,rows,fields in [('coverage',coverage,['task_id','status']),('dimension_ratings',dims,['task_id','option']),('dimension_ratings_from_invalid_responses',partial,['task_id','option']),('repair_failure_records',failures,['task_id','status'])]:metrics.csvwrite(directory/(name+'.csv'),rows,fields=None if rows else fields)
 write(directory/'predictions.json',dict(model=cfg,source_manifest_sha256=manifest['source_manifest_sha256'],predictions=predictions))
 if (directory/'run_manifest.json').exists():
  r=read(directory/'run_manifest.json');metrics.csvwrite(directory/'model_versions.csv',[dict(model_key=cfg['key'],model_id=cfg['repo_id'],revision=cfg['revision'],dtype=cfg['dtype'],thinking=cfg['enable_thinking'],backend=cfg['backend'],backend_version=cfg['backend_version'],run_id=r['run_id'],started_at=r['started_at'],completed_at=r.get('completed_at'),generation=cfg['generation'])])

def analyze(args):
 manifest,_=verify();reference=read(ROOT/'reference/human_consensus.json');assert reference['source_manifest_sha256']==manifest['source_manifest_sha256']
 refs={t:baseline.canonical_dense_ranks(g) for t,g in reference['task_rankings'].items()}
 allrows=[];per_task=[]
 for directory in sorted((ROOT/'runs').glob('*')):
  if not (directory/'run_manifest.json').exists():continue
  cfg=read(directory/'run_manifest.json')['identity']['config'];export(argparse.Namespace(model=cfg['key']))
  pred=read(directory/'predictions.json');assert pred['source_manifest_sha256']==manifest['source_manifest_sha256']
  rows={p['task_id']:metrics.metric(baseline.canonical_dense_ranks(p['overall_ranking']),refs[p['task_id']]) for p in pred['predictions']}
  per_task.extend(dict(model_key=cfg['key'],task_id=t,**r) for t,r in rows.items())
  allrows.append(dict(model_key=cfg['key'],model_name=cfg['display_name'],comparison_status='complete_scope' if len(rows)==30 else 'partial_coverage_only',missing_or_invalid_tasks=30-len(rows),**metrics.summarize(rows,manifest['formal_task_ids'])))
 assert allrows,'No formal runs present'
 allrows.sort(key=lambda r:(r['comparison_status']!='complete_scope',-(r['mean_task_tau_b'] if r['mean_task_tau_b'] is not None else -2)))
 dest=ROOT/'analysis';dest.mkdir(exist_ok=True)
 metrics.csvwrite(dest/'rgb30_bf16_results.csv',allrows);metrics.csvwrite(dest/'per_task_metrics.csv',per_task,fields=None if per_task else ['model_key','task_id','tau_b'])
 metrics.latex(dest/'rgb30_bf16_results.tex','rgb30_expanded',allrows)
 (dest/'summary.md').write_text('# RGB30 BF16 configurations\n\n'+metrics.table_md(allrows)+'\n\nRGB30 expanded: tasks 06/10/13 rely on author version equivalence. No independent per-participant view hashes. Task bootstrap: 10,000, seed 20260908, conditional on the fixed human reference. Partial coverage is descriptive only. Thinking/precision/backend changes are not an isolated quantization ablation.\n')
 write(dest/'analysis_manifest.json',dict(created_at=now(),source_manifest_sha256=manifest['source_manifest_sha256'],reference_sha256=sha(ROOT/'reference/human_consensus.json'),code_sha256=sha(Path(__file__)),bootstrap=dict(resamples=10000,seed=20260908,unit='task'),scope='rgb30_expanded'))
 print(dest/'rgb30_bf16_results.csv')

def pack(args):
 verify();names=[]
 for name in ['runs','analysis','deployment']:
  directory=ROOT/name
  if directory.exists():
   for p in directory.rglob('*'):
    if p.is_file() and p.suffix not in ['.lock','.tmp'] and p.name!='server.log':names.append(p)
 dest=ROOT/'rgb30_bf16_results.zip'
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(names):z.write(p,relative(p))
  for name in ['frozen_input_manifest.json','run_plan.json','bundle_manifest.json']:z.write(ROOT/name,name)
  for p in (ROOT/'configs').glob('*.json'):z.write(p,relative(p))
 print(dest)

def main():
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
 sub.add_parser('verify');sub.add_parser('analyze');sub.add_parser('pack-results')
 for name in ['serve','smoke','run','export']:
  p=sub.add_parser(name);p.add_argument('--model',required=True)
  if name=='serve':p.add_argument('--gpus',required=True);p.add_argument('--port',type=int,default=8000)
 args=parser.parse_args()
 if args.command=='verify':m,p=verify();print('PASS: 30 frozen inputs and bundle hashes; no model calls.')
 else:{'serve':serve,'smoke':smoke,'run':run,'export':export,'analyze':analyze,'pack-results':pack}[args.command](args)
if __name__=='__main__':main()
