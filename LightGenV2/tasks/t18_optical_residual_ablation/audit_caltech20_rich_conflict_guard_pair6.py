"""Independently check each TRAIN-task derivative and conflict-removal projection."""
import argparse,os,inspect,types,hashlib,numpy as np,torch
from pathlib import Path
import audit_caltech20_rich_confusion_pair_readout46 as previous
import caltech20_rich_conflict_guard_pair6 as m
saved=previous.saved
f=previous.independent_plan;independent_plan=types.FunctionType(f.__code__,globals(),'independent_pair_sampler',f.__defaults__,f.__closure__)
EVENTS=[]
def independent_gradient(x,y,target,w,b,w0,b0,pair):
 a,k=pair;tasks=[];active=[]
 for n in range(4):
  xx=x[n*16:(n+1)*16];yy=y[n*16:(n+1)*16];q=target[n*16:(n+1)*16];z=xx@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);r=z.copy();r[np.arange(16),yy]=-np.inf;j=r.argmax(1);hng=np.maximum(0,1+z[np.arange(16),j]-z[np.arange(16),yy]);dz=2*(p-q);dz[np.arange(16),j]+=2*hng;dz[np.arange(16),yy]-=2*hng;dz/=16;gg=dz.T@xx+.2*(w-w0);hh=dz.sum(0)+.2*(b-b0);g=np.zeros_like(gg);h=np.zeros_like(hh);g[a]=(gg[a]-gg[k])/2;g[k]=-g[a];h[a]=(hh[a]-hh[k])/2;h[k]=-h[a];tasks.append((g,h));active.append(float((hng>0).mean()))
 ng,nh=tasks[0];norm=float((ng*ng).sum()+(nh*nh).sum());corrected=[tasks[0]];events=[]
 for g,h in tasks[1:]:
  dot=float((g*ng).sum()+(h*nh).sum());factor=min(0.,dot)/(norm+1e-20);g=g-factor*ng;h=h-factor*nh;after=float((g*ng).sum()+(h*nh).sum());assert after>=-1e-9;events.append(dict(conflict=dot<0,dot_before=dot,dot_after=after));corrected.append((g,h))
 EVENTS.append(dict(pair=list(pair),neighbor_gradient_norm=float(np.sqrt(norm)),replay=events));return sum(g for g,h in corrected)/4,sum(h for g,h in corrected)/4,float(np.mean(active))
def cpu():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_per_task_gradient_and_conflict_projection');d=6;q,pt,pred=m.incumbent(d);w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy();pl=m.plan(d,1,pred,w,b,[]);assert pl==independent_plan(d,1,pred,w,b,[]);ids=np.array(pl['batches'][0]);x=pred['train']['features'][ids].astype(float);y=pred['train']['labels'][ids];target=m.probabilities(x,w,b);errors=[]
 for n in range(4):
  sl=slice(n*16,(n+1)*16);xx=x[sl];yy=y[sl];t=target[sl];ww=torch.tensor(w,requires_grad=True);bb=torch.tensor(b,dtype=torch.float64,requires_grad=True);z=torch.tensor(xx)@ww.T+bb;other=z.clone();other[torch.arange(16),torch.tensor(yy)]=-torch.inf;loss=torch.relu(1+other.max(1).values-z[torch.arange(16),torch.tensor(yy)]).square().mean()+2*(-torch.tensor(t)*torch.log_softmax(z,1)).sum(1).mean();gg,hh=torch.autograd.grad(loss,(ww,bb));g,h,_=m.basic_gradient(xx,yy,t,w,b,w,b);err=max(float(np.max(abs(g-gg.numpy()))),float(np.max(abs(h-hh.numpy()))));assert err<1e-10;errors.append(err)
 a,c=m.gradient(x,y,target,w,b,w,b),independent_gradient(x,y,target,w,b,w,b,pl['pair']);assert np.max(abs(a[0]-c[0]))<1e-10 and np.max(abs(a[1]-c[1]))<1e-10 and a[2]==c[2];assert not set(pred['train']['ids'])&set(pred['test']['ids']);m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,parent=q,pair=pl['pair'],per_task_gradient_maxabs=errors,independent_projection_passed=True,real_optical_models=0,optical_forward=0,TEST_forward=0,optimizer_steps=0),exclusive=True)
text=inspect.getsource(previous.audit);assert text.count('for d in (4,6):')==1;text=text.replace('def audit():','def audit_kernel():',1).replace('for d in (4,6):','for d in (6,):')
start=text.index('z=xx@w.astype(');end=text.index('clip=min(1.,1/(',start)
text=text[:start]+"g,h,active=independent_gradient(xx,yy,t[ids],w,b,w0,b0,(a,k));"+text[end:]
text=text.replace("float((hinge>0).mean())-row['active_fraction']","active-row['active_fraction']")
exec(compile(text,__file__+':same_complete_pair_audit_independent_task_projection','exec'),globals())
def audit():
 audit_kernel();f=m.OUT/'L6';records=m.s.read(f/'gradient_conflicts.json');assert records==EVENTS;fit=m.s.read(f/'fit_result.json');assert len(records)==fit['actual_updates'] and fit['replay_conflict_events']==sum(z['conflict'] for r in records for z in r['replay']);q=m.s.read(m.OUT/'science/audit.json');q.update(independent_per_task_projection_replayed=True,replay_conflict_events=fit['replay_conflict_events'],L4_not_loaded_or_trained=True);m.s.save(m.OUT/'science/audit.json',q)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
