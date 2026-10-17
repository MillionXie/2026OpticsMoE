"""TRAIN-evidenced stronger anchor for the same5140 head; optical70 core fixed."""
import ast,inspect
import caltech20_rich_pair50_same_head as previous
for name,value in vars(previous).items():
 if not name.startswith('__') and not inspect.isfunction(value):globals()[name]=value
FILES=('caltech20_rich_pair50_anchored_head.py', 'caltech20_rich_pair50_anchored_head_profile.json', 'check_caltech20_rich_pair50_anchored_head.py', 'audit_caltech20_rich_pair50_anchored_head.py', 'CALTECH20_RICH_PAIR50_ANCHORED_HEAD_PROTOCOL_20261017.md')
CPU=HERE/'runs/smoke/caltech20_rich_pair50_anchored_head_cpu_20261017'
for node in ast.parse(inspect.getsource(previous)).body:
 if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','folder','admit'):exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())

def config():
 c=previous.config();c.update(s.read(HERE/FILES[1]));assert c['anchor_scale']==10 and c['new_inference_parameters']==0;return c

def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z

def folder(key):assert key in config()['cases'];return HERE/'runs/simulation'/config()['run']/key

def admit(key):
 arm,fit=previous.admit(key);old=previous.folder(key);meta=s.read(old/'metadata.json');assert meta['sources']==previous.sources() and meta['config']==previous.config();engine.verify_sources(meta['sources'],meta['git_commit'])
 for receipt,where in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
  z=s.read(old/receipt);assert z['actual_audit_returncode']==0 and z['audit_sha256']==s.sha(old/where/'audit.json') and s.read(old/where/'audit.json')['passed']
 for n in ('producer_exit.json','campaign_exit.json','test_exit.json'):
  z=s.read(old/n);assert z['returncode']==0 and z['actual_exit'] and gone(z['pid'])
 f=s.read(old/'fit/fit_result.json');assert f['candidate_metrics']['train']['accuracy']>.95 and f['candidate_metrics']['train']['accuracy']-f['candidate_metrics']['val']['accuracy']>.15,'No evidence for stronger head regularization'
 return arm,fit

text=previous.objective_text;assert 'penalty=(theta-anchor).square().sum()/(2*n)' in text;text=text.replace('penalty=(theta-anchor).square().sum()/(2*n)',"penalty=cfg['anchor_scale']*(theta-anchor).square().sum()/(2*n)");exec(compile(text,__file__+':objective','exec'),globals())
exec(compile(previous.fit_text,__file__+':fit','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test','group'],required=True);p.add_argument('--case');a=p.parse_args()
 try:globals()[a.phase](a.case)
 except BaseException as e:
  out=folder(a.case) if a.case else HERE/'runs/simulation'/config()['run'];mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
