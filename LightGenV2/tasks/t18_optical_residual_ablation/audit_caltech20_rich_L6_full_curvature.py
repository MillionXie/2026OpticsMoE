"""Independent full-Hessian reference, residual checks and complete scalar trajectory."""
import types,inspect,numpy as np,torch,argparse
import audit_caltech20_rich_L6_binary_rest as previous
import caltech20_rich_L6_full_curvature as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('m','previous'):globals()[n]=v
for name in ('independent_plan',):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=previous.independent_gradient;raw_independent=types.FunctionType(f.__code__,globals(),'independent_raw_binary_derivative',f.__defaults__,f.__closure__)
def independent_gradient(x,target,reference,q,u,v,u0,v0,family):
 g,h,stats,diagonal,bd=raw_independent(x,target,reference,q,u,v,u0,v0,'binary_rest');c=m.PROFILE;d=x@u.astype(np.float32).astype(float)+float(np.float32(v))-reference;p=1/(1+np.exp(-np.clip(d,-60,60)));soft=1/(1+np.exp(-np.clip(d/c['temperature'],-60,60)));weights=np.ones(len(x));weights[:16]=c['neighbor_weight'];cur=(weights*soft*(1-soft)+c['KL']*p*(1-p))/len(x);A=np.c_[x,np.ones(len(x))];B=A*np.sqrt(cur)[:,None];raw=np.r_[g,h];alpha=c['anchor'];direction=raw/alpha-B.T@np.linalg.solve(np.eye(len(x))+B@B.T/alpha,B@raw)/(alpha*alpha);H=alpha*np.eye(len(raw))+(A.T*cur)@A;assert np.max(abs(H@direction-raw))<1e-10;stats.update(preconditioned=True,full_curvature=True);return direction[:-1],float(direction[-1]),stats,diagonal,bd
def independent_update(u,v,mu,mv,g,h,u0,v0,family):
 c=m.PROFILE
 if family=='binary_rest_full_curvature_trust':u=u-c['trust_step']*g;v=v-c['trust_step']*h;mu,mv=np.zeros_like(mu),0.
 else:
  scale=min(1.,1/(np.sqrt((g*g).sum()+h*h)+1e-6));mu=c['momentum']*mu+g*scale;mv=c['momentum']*mv+h*scale;u-=c['lr']*mu;v-=c['lr']*mv
 scale=min(1.,c['row_radius']/max(np.sqrt(((u-u0)**2).sum()+(v-v0)**2),1e-12));return u0+(u-u0)*scale,float(v0+(v-v0)*scale),mu,mv
text=inspect.getsource(previous.audit);start=text.index('scale=min(1.,1/(');end=text.index('ww,bb=w.copy(),b.copy()',start);text=text[:start]+"u,v,mu,mv=independent_update(u,v,mu,mv,g,h,u0,v0,spec['family']);"+text[end:];exec(compile(text,__file__+':same_full_audit_with_explicit_Newton_trust_update','exec'),globals())
def cpu():
 assert m.CASE==0 and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_full_Hessian_solve_reference');m.frozen();q=m.initial_parent();pt,pred=m.load_parent(q);w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy();pl=m.plan(1,pred,w,b,[]);assert pl==independent_plan(1,pred,w,b,[]);label=pl['label'];x,target,ref,teacher=m.prepared(pred,w,b,label);ix=np.array(pl['batches'][0]);xx=x[ix];t=target[ix];rr=ref[ix];qq=teacher[ix];u=w[label].copy();v=float(b[label]);vec=torch.tensor(np.r_[u,v],dtype=torch.float64,requires_grad=True);weights=np.ones(64);weights[:16]=m.PROFILE['neighbor_weight'];c=m.PROFILE
 def objective(uv):
  d=torch.tensor(xx)@uv[:-1]+uv[-1]-torch.tensor(rr);loss=(torch.tensor(weights)*torch.nn.functional.binary_cross_entropy_with_logits(d/c['temperature'],torch.tensor(t),reduction='none')).mean()*c['temperature']**2+c['KL']*torch.nn.functional.binary_cross_entropy_with_logits(d,torch.tensor(qq));return loss+c['anchor']/2*((uv-torch.tensor(np.r_[u,v])).square().sum())
 raw=torch.autograd.grad(objective(vec),vec)[0].detach().numpy();H=torch.autograd.functional.hessian(objective,vec,vectorize=True).detach().numpy();reference=np.linalg.solve(H,raw);g,h,stats=m.gradient(xx,t,rr,qq,u,v,u,v,c['families'][0]);ig,ih,istat,_,_=independent_gradient(xx,t,rr,qq,u,v,u,v,c['families'][0]);error=max(float(np.max(abs(np.r_[g,h]-reference))),float(np.max(abs(np.r_[ig,ih]-reference))));assert error<1e-10 and stats==istat and float(raw@np.r_[g,h])>0;steps={}
 for family in c['families']:
  a=m.update_kernel(u.copy(),v,np.zeros_like(u),0.,g,h,u,v,family);b=independent_update(u.copy(),v,np.zeros_like(u),0.,ig,ih,u,v,family);assert np.max(abs(a[0]-b[0]))<1e-10 and abs(a[1]-b[1])<1e-10;assert np.linalg.norm(np.r_[a[0]-u,a[1]-v])<=c['row_radius']+1e-12;steps[family]=float(objective(torch.tensor(np.r_[a[0],a[1]])))-float(objective(vec))
 assert steps['binary_rest_full_curvature_trust']<0;assert not set(pred['train']['ids'])&set(pred['test']['ids']);m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,parent=q,label=label,full_Hessian_direction_maxabs=error,full_Hessian_condition=float(np.linalg.cond(H)),first_TRAIN_objective_changes=steps,real_optical_models=0,optical_forward=0,TEST_forward=0,optimizer_steps=0),exclusive=True);m.frozen()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
