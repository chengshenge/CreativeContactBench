import csv,json,statistics
import holistic_alignment_baseline as baseline
from jsonschema import Draft202012Validator,ValidationError
from container_repair_source import repair_json_container
def csvwrite(path,rows,fields=None):
 if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else '' if v is None else v for k,v in r.items()})

def mean(vals):return statistics.fmean(vals) if vals else None

def partial_ratings(raw,schema):
 # Preserve valid E/F/C fields separately when the direct ranking is invalid.
 # This never changes coverage or supplies a holistic ranking.
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('duplicate_json_key')
   out[k]=v
  return out
 try:
  candidate,method=repair_json_container(raw);obj=json.loads(candidate,object_pairs_hook=unique)
  ratings=obj['ratings']
  Draft202012Validator(dict(schema['properties']['ratings'],**{'$defs':schema['$defs']})).validate(ratings)
 except (ValueError,KeyError,TypeError,ValidationError):
  return None
 totals={o:sum(ratings[o].values()) for o in 'ABCD'}
 return ratings,totals,[[o for o in 'ABCD' if totals[o]==v] for v in sorted(set(totals.values()),reverse=True)]

def metric(own,ref):
 ht=baseline.top_group(ref);ot=baseline.top_group(own);eligible=len(ht)==1
 return dict(tau_b=baseline.kendall_tau_b(own,ref),top1_eligible=eligible,top1_hit=bool(ht&ot) if eligible else None,strict_top1_hit=(ot==ht and len(ot)==1) if eligible else None,tied_top=len(ot)>1,all_tied=len(set(own.values()))==1)

def summarize(rows,scope_ids):
 taus=[r['tau_b'] for r in rows.values() if r['tau_b'] is not None];eligible=[r for r in rows.values() if r['top1_eligible']]
 ci=baseline.bootstrap_model(rows,{t:t for t in rows},10000,20260908) if rows else {k:None for k in ['tau_ci_low','tau_ci_high','top1_ci_low','top1_ci_high','strict_top1_ci_low','strict_top1_ci_high']}
 return dict(mean_task_tau_b=mean(taus),**ci,n_scope_tasks=len(scope_ids),n_valid_tasks=len(rows),tau_defined_tasks=len(taus),tau_undefined_tasks=len(rows)-len(taus),tau_defined_task_ids=[t for t,r in rows.items() if r['tau_b'] is not None],top1_denominator=len(eligible),strict_top1_hits=sum(r['strict_top1_hit'] for r in eligible),strict_top1_agreement=mean([float(r['strict_top1_hit']) for r in eligible]),inclusive_top1_hits=sum(r['top1_hit'] for r in eligible),inclusive_top1_agreement=mean([float(r['top1_hit']) for r in eligible]),tied_top_tasks=sum(r['tied_top'] for r in rows.values()),all_tied_tasks=sum(r['all_tied'] for r in rows.values()),tied_top_frequency=sum(r['tied_top'] for r in rows.values())/len(rows) if rows else None,all_tied_frequency=sum(r['all_tied'] for r in rows.values())/len(rows) if rows else None,valid_task_ids=list(rows))

def f(value,digits=6):return 'NA' if value is None else f'{value:.{digits}f}'

def table_md(rows):
 lines=['| Model | Status | Valid / scope | Mean tau-b [95% CI] | Defined tau n | Strict Top-1 | Inclusive Top-1 | Top ties / all ties |','|---|---|---:|---|---:|---|---|---:|']
 for r in rows:lines.append(f"| {r['model_name']} | {r['comparison_status']} | {r['n_valid_tasks']}/{r['n_scope_tasks']} | {f(r['mean_task_tau_b'])} [{f(r['tau_ci_low'])}, {f(r['tau_ci_high'])}] | {r['tau_defined_tasks']} | {hit(r,'strict')} | {hit(r,'inclusive')} | {r['tied_top_tasks']} / {r['all_tied_tasks']} |")
 return '\n'.join(lines)

def latex(path,scope,rows):
 esc=baseline.latex_escape
 expanded=scope=='rgb30_expanded'
 caption=('RGB30 expanded holistic agreement under the registered author version-equivalence assumption for Tasks 06, 10, and 13. These tasks lack independent participant-view version tracing.' if expanded else 'RGB27 strict version-compatibility comparison, excluding Tasks 06, 10, and 13, using new RGB-only responses and the ten-participant holistic reference.')
 lines=[r'\begin{table*}[t]',r'\centering',r'\scriptsize',r'\caption{'+caption+'}',r'\label{tab:'+scope.replace('_','-')+'}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Model & Valid/$N$ & Mean $\tau_b$ [95\% CI] & $n_{\tau}$ & Strict Top-1 & Inclusive Top-1 & Top/all ties \\',r'\midrule']
 for r in rows:
  marker=r'$^{\dagger}$' if r['comparison_status']!='complete_scope' else ''
  def cell(prefix):
   val=r[prefix+'_top1_agreement'];return str(r[prefix+'_top1_hits'])+'/'+str(r['top1_denominator'])+(f' ({val*100:.1f}'+r'\%)' if val is not None else '')
  lines.append(f"{esc(r['model_name'])}{marker} & {r['n_valid_tasks']}/{r['n_scope_tasks']} & {f(r['mean_task_tau_b'],3)} [{f(r['tau_ci_low'],3)}, {f(r['tau_ci_high'],3)}] & {r['tau_defined_tasks']} & {cell('strict')} & {cell('inclusive')} & {r['tied_top_tasks']}/{r['all_tied_tasks']} "+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.99\linewidth}\scriptsize',r'Notes: New inference only; canonical A--D; VLM rubric v1.2. CIs use 10,000 task-bootstrap resamples (seed 20260908), conditional on the existing ten participants. Undefined task-level $\tau_b$ is omitted, never filled; $n_{\tau}$ may differ across models. Top-1 excludes human-reference top ties; strict requires a unique model top, inclusive accepts its top group. Top/all ties are counts among valid responses. $^{\dagger}$Incomplete coverage is descriptive only and is not ranked with complete-scope models; its valid-task subset and denominators differ. Human LOO is reported separately and is not an absolute ceiling.',r'\end{minipage}',r'\end{table*}'])
 path.write_text('\n'.join(lines)+'\n')

def hit(row,prefix):
 n=row['top1_denominator'];num=row[prefix+'_top1_hits'];value=row[prefix+'_top1_agreement'];return f'{num}/{n} ({100*value:.2f}%)' if value is not None else f'{num}/{n} (NA)'
