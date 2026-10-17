import sys,json,csv,unittest,tempfile,shutil,argparse
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ccb_rgb30 as r

class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.fixture=r.read(r.ROOT/'tests/historical_regression.json')['responses'][0]
  self.raw=self.fixture['raw_response']
 def test_frozen_input_equality(self):
  m,_=r.verify();self.assertEqual(len(m['tasks']),30);self.assertNotEqual(m['formal_task_ids'],['task-%02d'%i for i in range(1,31)])
 def test_direct_ranking_independent_of_scores(self):
  p,_,_,_=r.parse(self.raw);p['overall_ranking']=[['D'],['C'],['B'],['A']]
  for o in 'ABCD':
   for k in p['ratings'][o]:p['ratings'][o][k]=3
  out,_,_,derived=r.parse(json.dumps(p));self.assertEqual(out['overall_ranking'],p['overall_ranking']);self.assertEqual(derived,[list('ABCD')])
 def test_missing_ranking_never_rebuilt(self):
  p,_,_,_=r.parse(self.raw);p.pop('overall_ranking')
  with self.assertRaises(ValueError):r.parse(json.dumps(p))
 def test_repairs_only_container(self):
  p,_,_,_=r.parse(self.raw);raw=json.dumps(p)
  self.assertEqual(r.parse('```json\n'+raw+'\n```')[0],p)
  with self.assertRaises(ValueError):r.parse('Here is my answer: '+raw)
  with self.assertRaises(ValueError):r.parse(raw[:-1]+',"confidence":4}')
 def test_ties_undefined_and_top1(self):
  b=r.baseline;ref=b.canonical_dense_ranks([['A'],['B'],['C'],['D']])
  v=r.metrics.metric(b.canonical_dense_ranks([list('ABCD')]),ref)
  self.assertIsNone(v['tau_b']);self.assertFalse(v['strict_top1_hit']);self.assertTrue(v['top1_hit'])
  self.assertFalse(r.metrics.metric(ref,b.canonical_dense_ranks([['A','B'],['C'],['D']]))['top1_eligible'])
 def test_actual_request_contains_frozen_rgb_and_text_only(self):
  m,_=r.verify();t=m['tasks'][0];cfg=r.config('qwen3_8-27b-bf16')
  payload=r.body(cfg,(r.ROOT/t['prompt_path']).read_text(),(r.ROOT/t['image_path']).read_bytes(),'id')
  self.assertEqual(len(payload['messages']),1);self.assertEqual(len(payload['messages'][0]['content']),2)
  self.assertEqual(payload['messages'][0]['content'][0]['text'],(r.ROOT/t['prompt_path']).read_text())
  self.assertEqual(payload['chat_template_kwargs'],{'enable_thinking':True})
  self.assertNotIn('response_format',payload)
 def test_failure_classification(self):
  self.assertEqual(r.classify({'choices':[{'message':{'refusal':'no'}}]})['status'],'refusal')
  self.assertEqual(r.classify({'choices':[{'message':{'content':''},'finish_reason':'length'}]})['status'],'truncated')
  for status in ['valid','parse_failure','missing_fields','refusal','truncated','http_401','http_404']:self.assertFalse(r.retryable(status))
 def test_historical_metrics_and_bootstrap_exact(self):
  m,_=r.verify();refs=r.read(r.ROOT/'reference/human_consensus.json')['task_rankings'];rows={}
  for rec in r.read(r.ROOT/'tests/historical_regression.json')['responses']:
   p,_,_,_=r.parse(rec['raw_response']);rows.setdefault(rec['model_key'],{})[rec['task_id']]=r.metrics.metric(r.baseline.canonical_dense_ranks(p['overall_ranking']),r.baseline.canonical_dense_ranks(refs[rec['task_id']]))
  expected={x['model_key']:x for x in csv.DictReader((r.ROOT/'tests/expected_historical_results.csv').read_text().splitlines())}
  for key,v in rows.items():
   # Preserve canonical input order used by the original bootstrap.
   actual=r.metrics.summarize({t:v[t] for t in m['formal_task_ids']},m['formal_task_ids'])
   for field in ['mean_task_tau_b','tau_ci_low','tau_ci_high','strict_top1_hits','inclusive_top1_hits','top1_denominator','tau_defined_tasks','tied_top_tasks','all_tied_tasks','strict_top1_ci_low','strict_top1_ci_high','top1_ci_low','top1_ci_high']:
    self.assertAlmostEqual(actual[field],float(expected[key][field]),places=12,msg=key+' '+field)
 def test_end_to_end_resume_and_retry_cap(self):
  original=r.ROOT
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'bundle';shutil.copytree(original,target,ignore=shutil.ignore_patterns('__pycache__','runs','deployment','analysis'))
   with patch.object(r,'ROOT',target):
    cfg=r.config('qwen3_8-27b-bf16');state=dict(session_dir='deployment/mock',served_model_id='mock',port=8000)
    r.write(target/'deployment/mock/smoke/smoke_result.json',{'passed':True})
    calls=[]
    def fake(cfg,state,payload,dest):
     calls.append(dest);tid=dest.parent.name
     if tid in ['task-01','task-02','task-03','task-04']:return dict(status='timeout',valid=False)
     (dest/'raw_response.txt').write_text(self.raw)
     result=r.classify({'choices':[{'message':{'content':self.raw}}]});return result
    args=argparse.Namespace(model=cfg['key'])
    with patch.object(r,'state_for',return_value=state),patch.object(r,'request',side_effect=fake),patch.object(r.time,'sleep'):
     r.run(args);self.assertEqual(len(calls),33)
     r.run(args);self.assertEqual(len(calls),33)
    fourth=r.read(target/'runs'/cfg['key']/'tasks/task-04/result.json')
    self.assertTrue(fourth['selected_attempt_path'].endswith('attempt-01'))
    r.analyze(args);r.pack(args)
    self.assertTrue((target/'rgb30_bf16_results.zip').exists())
 def test_tamper_detection(self):
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'bundle';shutil.copytree(r.ROOT,target,ignore=shutil.ignore_patterns('__pycache__','runs','deployment','analysis'))
   with patch.object(r,'ROOT',target):
    (target/'inputs/task-01/prompt.txt').write_text('changed')
    with self.assertRaises(AssertionError):r.verify()

if __name__=='__main__':unittest.main()
