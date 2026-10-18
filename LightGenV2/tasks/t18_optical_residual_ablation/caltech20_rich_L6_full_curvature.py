"""L6 only: exact local scalar-head curvature via a64x64 Woodbury solve."""
import types,numpy as np
import caltech20_rich_L6_binary_rest as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_L6_full_curvature.py','caltech20_rich_L6_full_curvature_profile.json','audit_caltech20_rich_L6_full_curvature.py','CALTECH20_RICH_L6_FULL_CURVATURE_PROTOCOL_20261018.md')
PROFILE=s.read(HERE/FILES[1]);CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke/caltech20_rich_L6_full_curvature_CPU_20261018'
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def initial_parent():
 completed=s.read(previous.ROOT_OUT/'completed.json');ex=s.read(previous.ROOT_OUT/'campaign_exit.json');assert completed['passed'] and ex['actual_exit'] and ex['returncode']==0;q=dict(completed['best']);assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
for name in ('identity','claim','specification','config','frozen','load_parent','ready','campaign','plan','prepared','fit'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=previous.gradient;raw_gradient=types.FunctionType(f.__code__,globals(),'same_direct_binary_objective_gradient',f.__defaults__,f.__closure__)
f=previous.update;original_update=types.FunctionType(f.__code__,globals(),'same_original_scale_update',f.__defaults__,f.__closure__)
def update_kernel(u,v,mu,mv,g,h,u0,v0,family):
 if family!='binary_rest_full_curvature_trust':return original_update(u,v,mu,mv,g,h,u0,v0)
 u=u-PROFILE['trust_step']*g;v=v-PROFILE['trust_step']*h;scale=min(1.,PROFILE['row_radius']/max(np.sqrt(((u-u0)**2).sum()+(v-v0)**2),1e-12));return u0+(u-u0)*scale,float(v0+(v-v0)*scale),np.zeros_like(mu),0.
def update(u,v,mu,mv,g,h,u0,v0):return update_kernel(u,v,mu,mv,g,h,u0,v0,specification()['family'])
def gradient(x,target,reference,q,u,v,u0,v0,family):
 c=PROFILE;g,h,stats=raw_gradient(x,target,reference,q,u,v,u0,v0,'binary_rest');d=x@u.astype(np.float32).astype(float)+float(np.float32(v))-reference;p=1/(1+np.exp(-np.clip(d,-60,60)));soft=1/(1+np.exp(-np.clip(d/c['temperature'],-60,60)));weights=np.ones(len(x));weights[:16]=c['neighbor_weight'];cur=(weights*soft*(1-soft)+c['KL']*p*(1-p))/len(x);A=np.c_[x,np.ones(len(x))];B=A*np.sqrt(cur)[:,None];raw=np.r_[g,h];alpha=c['anchor'];direction=raw/alpha-B.T@np.linalg.solve(np.eye(len(x))+B@B.T/alpha,B@raw)/(alpha*alpha);stats.update(preconditioned=True,full_curvature=True);return direction[:-1],float(direction[-1]),stats
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:fit() if a.phase=='fit' else campaign()
 except BaseException as e:
  folder=OUT if a.phase=='fit' else ROOT_OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
