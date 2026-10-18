"""Continue verified341/335 heads across remaining error classes; same inference graph."""
import types
import caltech20_rich_local_row_scalar_reference46 as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_local_row_coverage46.py','caltech20_rich_local_row_coverage46_profile.json','audit_caltech20_rich_local_row_coverage46.py','CALTECH20_RICH_LOCAL_ROW_COVERAGE46_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_local_row_coverage46_CPU_20261018'
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def incumbent(d):
 rec=s.read(previous.OUT/'audit_completed.json');assert rec['passed'] and rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(previous.OUT/'science/audit.json')
 f=previous.OUT/f'L{d}';p=f/'best.pt';q=dict(path=str(p),sha256=s.sha(p),source_science=rec['audit_sha256']);assert q['sha256']==config()['parents'][str(d)]
 pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(f/('selected_'+r+'.npz')) for r in ('train','val','test')};assert int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum())==config()['parent_correct'][str(d)]
 return q,pt,pred
for name in ('identity','claim','stage_plan','prepared','gradient','update','ready','fit'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
