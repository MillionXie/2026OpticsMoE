"""L6 only: remove replay components opposing the targeted TRAIN-neighbor gradient."""
import types,inspect
import caltech20_rich_pair_gain_continuation46 as closure
import caltech20_rich_confusion_pair_readout46 as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('previous','closure'):globals()[n]=v
FILES=('caltech20_rich_conflict_guard_pair6.py','caltech20_rich_conflict_guard_pair6_profile.json','audit_caltech20_rich_conflict_guard_pair6.py','CALTECH20_RICH_CONFLICT_GUARD_PAIR6_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_conflict_guard_pair6_CPU_20261018';ACTIVE_PAIR=None;GRADIENT_EVENTS=[]
def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def incumbent(d):
 assert d==6;root=closure.ROOT_OUT/'round01';rec=s.read(root/'audit_completed.json');assert rec['passed'] and rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(root/'science/audit.json');f=root/'L6';p=f/'best.pt';q=dict(path=str(p),sha256=s.sha(p),source_science=rec['audit_sha256']);assert q['sha256']==config()['parents']['6'];pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(f/('selected_'+r+'.npz')) for r in ('train','val','test')};return q,pt,pred
for name in ('identity','claim','projected','update'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=previous.plan;pair_plan=types.FunctionType(f.__code__,globals(),'same_pair_TRAIN_plan',f.__defaults__,f.__closure__)
def plan(d,stage,pred,w,b,used):
 global ACTIVE_PAIR
 q=pair_plan(d,stage,pred,w,b,used)
 if q is not None:ACTIVE_PAIR=tuple(q['pair'])
 return q
f=previous.gradient;basic_gradient=types.FunctionType(f.__code__,globals(),'verified_full_margin_task_gradient',f.__defaults__,f.__closure__)
def gradient(x,y,q,w,b,w0,b0):
 assert len(x)==64 and ACTIVE_PAIR is not None;tasks=[];active=[]
 for k in range(4):
  sl=slice(16*k,16*(k+1));g,h,a=basic_gradient(x[sl],y[sl],q[sl],w,b,w0,b0);g,h=projected(g,h,ACTIVE_PAIR);tasks.append((g,h));active.append(a)
 ng,nh=tasks[0];norm=float((ng*ng).sum()+(nh*nh).sum());events=[];corrected=[tasks[0]]
 for g,h in tasks[1:]:
  dot=float((g*ng).sum()+(h*nh).sum());factor=min(0.,dot)/(norm+1e-20);g=g-factor*ng;h=h-factor*nh;events.append(dict(conflict=dot<0,dot_before=dot,dot_after=float((g*ng).sum()+(h*nh).sum())));corrected.append((g,h))
 GRADIENT_EVENTS.append(dict(pair=list(ACTIVE_PAIR),neighbor_gradient_norm=float(np.sqrt(norm)),replay=events));return sum(g for g,h in corrected)/4,sum(h for g,h in corrected)/4,float(np.mean(active))
def ready():
 q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();parent.old.controls();incumbent(6)
text=inspect.getsource(previous.fit);assert text.count('for d in (4,6):')==1;text=text.replace('def fit():','def fit_kernel():',1).replace('for d in (4,6):','for d in (6,):');exec(compile(text,__file__+':same_pair_fit_L6_only','exec'),globals())
def fit():
 fit_kernel();f=OUT/'L6';s.save(f/'gradient_conflicts.json',GRADIENT_EVENTS,exclusive=True);q=s.read(f/'fit_result.json');assert len(GRADIENT_EVENTS)==q['actual_updates'];q['hashes']['gradient_conflicts.json']=s.sha(f/'gradient_conflicts.json');q['replay_conflict_events']=sum(z['conflict'] for r in GRADIENT_EVENTS for z in r['replay']);s.save(f/'fit_result.json',q)
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
