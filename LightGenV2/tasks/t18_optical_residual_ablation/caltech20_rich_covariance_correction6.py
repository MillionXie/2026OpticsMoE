"""Same TRAIN covariance correction, explicitly from new L6 334-correct head."""
import types
import caltech20_rich_covariance_correction4 as baseline
import caltech20_rich_margin_guard_readout46 as gained
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('baseline','closure','warm'):globals()[n]=v
FILES=('caltech20_rich_covariance_correction6.py','caltech20_rich_covariance_correction6_profile.json','check_caltech20_rich_covariance_correction6.py','audit_caltech20_rich_covariance_correction6.py','CALTECH20_RICH_COVARIANCE_CORRECTION6_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_covariance_correction6_CPU_20261018'
def sources():
 z=baseline.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def incumbent():
 rec=s.read(gained.OUT/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(gained.OUT/'science/audit.json');f=gained.OUT/'L6';path=f/'best.pt';assert s.sha(path)==config()['parent_SHA256'];pt=torch.load(path,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(f/('selected_'+r+'.npz')) for r in ('train','val','test')};return dict(path=str(path),sha256=s.sha(path),source_science=rec['audit_sha256'],parent_head_step=24),pt,pred
for name in ('identity','claim','statistics','correction','ready','fit'):
 f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
