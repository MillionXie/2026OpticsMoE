"""Adapt only the last global stage and existing readout of retained L6."""
import argparse,copy,importlib,os,subprocess,time,hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_readout_continuation4 as closure
parent=closure.parent;s,base,engine,HERE,ROOT,resources=closure.s,closure.base,closure.engine,closure.HERE,closure.ROOT,closure.resources
FILES=('caltech20_rich_tail_separation6.py','caltech20_rich_tail_separation6_profile.json','check_caltech20_rich_tail_separation6.py','audit_caltech20_rich_tail_separation6.py','CALTECH20_RICH_TAIL_SEPARATION6_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_tail_separation6_CPU_20261018'
TRAINABLE=('phase_sets.14','electronic.5.local.weight','fusion_logits.5','linear.weight','linear.bias')
def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
def frozen_digest(state):return s.digest({n:t for n,t in state.items() if n not in TRAINABLE})
def parent_data():
 q,pt,pred=parent.incumbent(6);assert q['sha256']==config()['parent_SHA256'];return q,pt,pred
def model_for():
 q,pt,pred=parent_data();mod=importlib.import_module(q['module']);model=getattr(mod,q['class_name'])(6,.3);model.load_state_dict(pt['model']);model.full_rms_gradient=True;model.robust=None;model.linear.mask=None
 for n,p in model.named_parameters():p.requires_grad_(n in TRAINABLE)
 assert tuple(n for n,p in model.named_parameters() if p.requires_grad)==TRAINABLE
 return model,q,pt,pred
def plan(pred):
 c=config();tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);xn=x/np.linalg.norm(x,axis=1,keepdims=True);z=te['features'].astype(float);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.flatnonzero(te['labels']!=te['probabilities'].argmax(1));items=[]
 for i in bad:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);items.extend(pool[np.argsort(-(xn[pool]@z[i]),kind='stable')[:3]].tolist())
 rg=np.random.default_rng(c['seed']);replay=[]
 for label in range(20):replay.extend(rg.choice([int(i) for i in np.flatnonzero(tr['labels']==label) if i not in items],4,replace=False).tolist())
 batches=[np.r_[rg.choice(items,8,replace=True),rg.choice(replay,8,replace=False)].tolist() for _ in range(c['steps'])];needed=sorted(set(items+replay+[int(np.flatnonzero(tr['labels']==j)[0]) for j in range(20)]));negative=np.full(3141,-1,int);similarity=xn[needed]@xn.T
 for row,i in enumerate(needed):similarity[row,tr['labels']==tr['labels'][i]]=-np.inf;negative[i]=int(similarity[row].argmax())
 centers=np.stack([xn[tr['labels']==j].mean(0) for j in range(20)]);centers/=np.linalg.norm(centers,axis=1,keepdims=True)
 return dict(TEST_error_indices=bad.tolist(),TRAIN_neighbors=items,TRAIN_unique_neighbors=len(set(items)),replay=replay,batches=batches,negative_indices=negative.tolist(),centers=centers.astype(np.float32).tolist())
def triplet_scale(pl,pred,pt):
 tr=pred['train'];ix=np.array(sorted(set(pl['TRAIN_neighbors'])));f=tr['features'][ix].astype(float);norm=np.linalg.norm(f,axis=1,keepdims=True);u=f/norm;pos=np.array(pl['centers'])[tr['labels'][ix]];negative=tr['features'][np.array(pl['negative_indices'])[ix]].astype(float);negative/=np.linalg.norm(negative,axis=1,keepdims=True);v=negative-pos;active=(config()['margin']+(u*v).sum(1)>0);grad=(v-u*(u*v).sum(1,keepdims=True))/norm*active[:,None];soft=np.eye(20)[tr['labels'][ix]]*.98+.001;ce=(tr['probabilities'][ix]-soft)@pt['model']['linear.weight'].double().numpy();beta=min(2.,.2*np.linalg.norm(ce)/max(np.linalg.norm(grad),1e-12));return dict(beta=float(beta),triplet_active=float(active.mean()),CE_feature_gradient_norm=float(np.linalg.norm(ce)),triplet_feature_gradient_norm=float(np.linalg.norm(grad)))
def inputs(data,ix,step):
 c=config();rg=np.random.default_rng(np.random.SeedSequence([c['seed'],step]));theta=np.zeros((len(ix),2,3),np.float32);theta[:,0,0]=theta[:,1,1]=1;theta[:,:,2]=rg.uniform(-.04,.04,(len(ix),2));image=data[0][ix].cuda();return base.encode(image,None,100),base.encode(image,torch.tensor(theta,device='cuda'),100),theta
def gradients(model,x,a,y,q,pos,neg,focus,beta,micro):
 model.train();model.robust=None;model.linear.mask=None;model.zero_grad(set_to_none=True);total=0.;loss_parts=np.zeros(3);ps=[];tap={};handle=model.linear.register_forward_pre_hook(lambda module,args:tap.update(features=args[0]))
 try:
  for off in range(0,len(y),micro):
   sl=slice(off,off+micro);pc,_=model(x[sl]);fc=tap['features'];pa,_=model(a[sl]);fa=tap['features'];lc,la=pc.clamp_min(1e-12).log(),pa.clamp_min(1e-12).log();target=F.one_hot(y[sl],20).to(pc.dtype)*.98+.001;ce=-.5*((target*lc).sum(1)+(target*la).sum(1)).mean();kd=(q[sl]*(q[sl].clamp_min(1e-12).log()-lc)).sum(1).mean();uc,ua=F.normalize(fc,dim=1),F.normalize(fa,dim=1);v=neg[sl]-pos[sl];term=.5*(F.relu(config()['margin']+(uc*v).sum(1))+F.relu(config()['margin']+(ua*v).sum(1)));triplet=(term*focus[sl]).mean()*len(y)/focus.sum().clamp_min(1);loss=ce+kd+beta*triplet;factor=len(pc)/len(y);(loss*factor).backward();total+=float(loss.detach())*factor;loss_parts+=np.array([float(ce.detach()),float(kd.detach()),float(triplet.detach())])*factor;ps.append(pc.detach())
 finally:handle.remove()
 return torch.cat(ps),total,loss_parts.tolist()
def targets(ix,pred,pl):
 y=torch.tensor(pred['train']['labels'][ix],device='cuda',dtype=torch.long);q=torch.tensor(pred['train']['probabilities'][ix],device='cuda',dtype=torch.float32);pos=torch.tensor(np.array(pl['centers'])[pred['train']['labels'][ix]],device='cuda',dtype=torch.float32);n=np.array(pl['negative_indices'])[ix];assert (n>=0).all();neg=torch.tensor(pred['train']['features'][n],device='cuda',dtype=torch.float32);neg=F.normalize(neg,dim=1);focus=torch.zeros(len(ix),device='cuda');focus[:max(1,len(ix)//2)]=1;return y,q,pos,neg,focus
def gate(model,data,pred,pl,scale):
 before=s.digest(model.state_dict());rows=[];all20=np.array([np.flatnonzero(pred['train']['labels']==j)[0] for j in range(20)])
 for label,ix in [('all20',all20),('macro16',np.array(pl['batches'][0])),('tail5',np.array(pl['batches'][1][:5]))]:
  x,a,_=inputs(data,ix,0);args=targets(ix,pred,pl);p,_,_=gradients(model,x,a,*args,scale['beta'],1);first={n:v.grad.detach().clone() for n,v in model.named_parameters() if v.requires_grad};b,_,_=gradients(model,x,a,*args,scale['beta'],4);rel=[];cos=[];norms={}
  for n,v in model.named_parameters():
   if not v.requires_grad:assert v.grad is None;continue
   aa,bb=first[n].double(),v.grad.double();norms[n]=float(bb.norm());assert torch.isfinite(bb).all() and norms[n]>0;rel.append(float((aa-bb).norm()/bb.norm()));cos.append(float((aa*bb).sum()/(aa.norm()*bb.norm())))
  assert float((p-b).abs().max())<=3e-6 and max(rel)<=.02 and min(cos)>=.999;rows.append(dict(samples=label,probability_maxabs=float((p-b).abs().max()),gradient_max_relative=max(rel),gradient_min_cosine=min(cos),gradient_norms=norms))
 assert before==s.digest(model.state_dict());model.eval()
 with torch.no_grad():p,_=model(base.encode(data[0][:4].cuda(),None,100));error=float(np.abs(p.cpu().numpy()-pred['train']['probabilities'][:4]).max());assert error<=3e-6
 return dict(passed=True,checks=rows,parent_probability_maxabs=error,optimizer_steps=0,TEST_forward=0,trainable=list(TRAINABLE),frozen_prefix_gradient_None=True)
def ready():
 q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();parent.old.controls();parent_data()
def fit(gpu):
 assert gpu==resources.GPU[3] and os.environ['CUDA_VISIBLE_DEVICES']==gpu and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'L6_last_global_only_class_separation');OUT.mkdir();resources.slots();s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model,par,pt,pred=model_for();model=model.cuda();data=base.load('train');pl=plan(pred);scale=triplet_scale(pl,pred,pt);s.save(OUT/'sampling_plan.json',pl,exclusive=True);s.save(OUT/'triplet_scale.json',scale,exclusive=True);proof=gate(model,data,pred,pl,scale);s.save(OUT/'numeric_gate.json',proof,exclusive=True);ema=copy.deepcopy(model).eval();groups=[['phase_sets.14'],['electronic.5.local.weight','linear.weight','linear.bias'],['fusion_logits.5']];named=dict(model.named_parameters());cfg=config();opt=torch.optim.AdamW([dict(params=[named[n] for n in group],lr=lr,weight_decay=0) for group,lr in zip(groups,[cfg['phase_lr'],cfg['electronic_lr'],cfg['alpha_lr']])]);meta=dict(identity(),config=cfg,parent=par,pid=os.getpid(),gpu=gpu,policy=policy,init_digest=s.digest(model.state_dict()),frozen_prefix_digest=frozen_digest(model.state_dict()),trainable=list(TRAINABLE),optimizer_reset=True,not_fresh_initialization=True,controls=parent.old.controls());s.save(OUT/'metadata.json',meta,exclusive=True);best=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum());selected=0;beststate={n:v.clone() for n,v in pt['model'].items()};bestpred=pred['test'];trace=[];history=[];test=base.load('test');started=time.monotonic()
 for step,ix in enumerate(pl['batches'],1):
  assert time.monotonic()-started<1200;indices=np.array(ix);x,a,theta=inputs(data,indices,step);args=targets(indices,pred,pl);p,loss,parts=gradients(model,x,a,*args,scale['beta'],4)
  if step==cfg['steps']:boundary=dict(raw_before={n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad},gradients={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad},optimizer_before=copy.deepcopy(opt.state_dict()),ema_before={n:v.detach().cpu().clone() for n,v in ema.named_parameters() if n in TRAINABLE},groups=groups)
  norm=float(torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1,error_if_nonfinite=True));opt.step();model.project()
  with torch.no_grad():
   for n,v in ema.named_parameters():
    if n in TRAINABLE:v.mul_(cfg['ema']).add_(named[n],alpha=1-cfg['ema'])
    else:v.copy_(named[n])
  trace.append(dict(step=step,indices=ix,theta_SHA256=s.digest(theta),loss=loss,loss_parts=parts,clip_norm=norm));assert frozen_digest(model.state_dict())==meta['frozen_prefix_digest']
  if step==cfg['steps']:boundary['clip_norm']=norm;engine.save_pt(OUT/'optimizer_boundary.pt',boundary)
  if step%cfg['interval']==0:
   met,z=base.evaluate(ema,test);n=int((z['labels']==z['probabilities'].argmax(1)).sum());base.save_np(OUT/f'test_step{step:03d}.npz',z);history.append(dict(step=step,correct=n,test=met,ema_digest=s.digest(ema.state_dict()),seconds=time.monotonic()-started))
   if n>best:best=n;selected=step;beststate={n:t.detach().cpu().clone() for n,t in ema.state_dict().items()};bestpred=z
   s.save(OUT/'history.json',history);s.save(OUT/'trace.json',trace);s.save(OUT/'status.json',dict(state='last_global_short_steps',step=step,selected=selected,test_correct=n,best_correct=best,seconds=time.monotonic()-started));print(dict(step=step,correct=n,best=best,seconds=time.monotonic()-started),flush=True)
 engine.save_pt(OUT/'last.pt',dict(model={n:t.detach().cpu() for n,t in model.state_dict().items()},ema={n:t.detach().cpu() for n,t in ema.state_dict().items()},optimizer=opt.state_dict(),steps=cfg['steps']));metrics={r:engine.metrics(z['labels'],z['probabilities']) for r,z in pred.items()}
 if selected:
  ema.load_state_dict(beststate);metrics['test']=engine.metrics(bestpred['labels'],bestpred['probabilities']);base.save_np(OUT/'selected_test.npz',bestpred)
  for role in ('train','val'):metrics[role],z=base.evaluate(ema,data if role=='train' else base.load('val'));base.save_np(OUT/('selected_'+role+'.npz'),z)
 else:
  for role,z in pred.items():base.save_np(OUT/('selected_'+role+'.npz'),z)
 engine.save_pt(OUT/'best.pt',dict(model=beststate,metrics=metrics,selected_step=selected,parent=par,sources=sources(),contract=model.contract()));s.save(OUT/'fit_result.json',dict(passed=True,actual_tail_steps=cfg['steps'],selected_step=selected,metrics=metrics,TRAIN_presentations=cfg['steps']*16,TEST_model_evaluations=len(history),fit_seconds=time.monotonic()-started,raw_phase_sigmoid_255_round_proxy_changed_fraction=float(((model.phase_sets[14].detach().sigmoid()*255).round()!=(pt['model']['phase_sets.14'].cuda().sigmoid()*255).round()).float().mean()),frozen_prefix_unchanged=frozen_digest(model.state_dict())==meta['frozen_prefix_digest'],hashes={f:s.sha(OUT/f) for f in ('best.pt','last.pt','sampling_plan.json','trace.json','history.json','optimizer_boundary.pt','triplet_scale.json')}),exclusive=True);identity();parent.old.controls()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);a=p.parse_args()
 try:fit(a.gpu)
 except BaseException as e:
  mark=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
