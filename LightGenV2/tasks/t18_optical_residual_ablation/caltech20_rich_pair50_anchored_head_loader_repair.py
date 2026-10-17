"""Explicit callable-alias repair, unchanged factor10 head objective."""
import ast,inspect
import caltech20_rich_pair50_anchored_head as failed
for name,value in vars(failed).items():
 if not name.startswith('__') and not inspect.isfunction(value):globals()[name]=value
EXTRA=('caltech20_rich_pair50_anchored_head_loader_repair.py', 'check_caltech20_rich_pair50_anchored_head_loader_repair.py', 'audit_caltech20_rich_pair50_anchored_head_loader_repair.py', 'CALTECH20_RICH_PAIR50_ANCHORED_HEAD_LOADER_REPAIR_PROTOCOL_20261017.md')
FILES=(EXTRA[0],failed.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
CPU=HERE/'runs/smoke/caltech20_rich_pair50_anchored_head_loader_repair_cpu_20261017'
FAILURE_SHA='c0ebfc3ca9dcc399f0743809ff71e56acf26b82cefbc32ece4bf1f56c67a2516'
load_np=failed.previous.load_np
probability=failed.previous.probability
logits=failed.previous.logits
core_digest=failed.previous.core_digest
for node in ast.parse(inspect.getsource(failed.previous)).body:
 if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','folder','admit'):exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())

def config():
 c=failed.config();c.update(run=c['run']+'_loader_repair',storage_repair='Bind all four pure cached-array callable aliases explicitly; objective/values/optimizer/selection unchanged',preserved_failure_sha256=FAILURE_SHA,failed_source_commit='9e7b0fe3be58b727176c70c486aefe6e1bff0c17');return c

def sources():
 z=failed.sources()
 for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z

def folder(key):assert key in config()['cases'];return HERE/'runs/simulation'/config()['run']/key

def admit(key):
 arm,fit=failed.admit(key)
 for k in config()['cases']:
  out=failed.folder(k);assert s.sha(out/'failure.json')==FAILURE_SHA and not (out/'fit/fit_result.json').exists() and not (out/'once_test').exists();ex=s.read(out/'campaign_exit.json');assert ex['returncode']==1 and ex['actual_exit'] and gone(ex['pid'])
 return arm,fit

exec(compile(failed.text,__file__+':unchanged_objective','exec'),globals());exec(compile(failed.previous.fit_text,__file__+':unchanged_fit','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test','group'],required=True);p.add_argument('--case');a=p.parse_args()
 try:globals()[a.phase](a.case)
 except BaseException as e:
  out=folder(a.case) if a.case else HERE/'runs/simulation'/config()['run'];mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
