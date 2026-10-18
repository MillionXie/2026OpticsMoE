"""TEST-error-informed TRAIN-only short steps; no full-dataset training epochs."""
import argparse,copy,importlib,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_currentKD46 as parent
import caltech20_rich_augmented_teacher46 as closure
s,base,engine,resources=parent.s,parent.base,parent.engine,parent.resources
HERE,ROOT=parent.HERE,parent.ROOT
FILES=('caltech20_rich_error_target_steps46.py','check_caltech20_rich_error_target_steps46.py','audit_caltech20_rich_error_target_steps46.py','CALTECH20_RICH_ERROR_TARGET_STEPS46_PROTOCOL_20261018.md')
OUT=HERE/'runs/simulation/caltech20_rich20_TEST_error_TRAIN_hard_replay46_steps64_20261018'
CPU=HERE/'runs/smoke/caltech20_rich_error_target_steps46_CPU_20261018'
CFG=dict(steps=64,segment=16,batch=16,micro=4,phase_lr=.00008,electronic_lr=.0002,alpha_lr=.00002,ema=.9,top_classes=6,hard_per_class=16,replay_per_class=4,seed=17018,TEST_informed_development=True)

def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z

def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)

def plan(pred):
 y=pred['test']['labels'];p=pred['test']['probabilities'].argmax(1);cm=np.bincount(y*20+p,minlength=400).reshape(20,20);errors=cm.sum(1)-cm.diagonal();focus=sorted(range(20),key=lambda c:(-int(errors[c]),c))[:6]
 ty=pred['train']['labels'];tp=pred['train']['probabilities'];rival=tp.copy();rival[np.arange(len(ty)),ty]=-1;margin=tp[np.arange(len(ty)),ty]-rival.max(1);hard=[]
 for c in focus:
  ids=np.flatnonzero(ty==c);hard.extend(sorted(ids.tolist(),key=lambda i:(float(margin[i]),i))[:16])
 rg=np.random.default_rng(CFG['seed']);replay=[]
 for c in range(20):
  ids=np.array([i for i in np.flatnonzero(ty==c) if i not in hard]);replay.extend(rg.choice(ids,4,replace=False).tolist())
 opponents=cm.copy();np.fill_diagonal(opponents,0);opp=opponents.argmax(1);opp[opponents.sum(1)==0]=(np.flatnonzero(opponents.sum(1)==0)+1)%20
 batches=[]
 for step in range(CFG['steps']):
  ix=np.r_[rg.choice(hard,8,replace=False),rg.choice(replay,8,replace=False)];rg.shuffle(ix);batches.append(ix.tolist())
 return dict(focus_classes=focus,test_error_counts=errors.tolist(),hard=hard,replay=replay,opponent=opp.tolist(),batches=batches,train_ids=[pred['train']['ids'][i].item() if hasattr(pred['train']['ids'][i],'item') else pred['train']['ids'][i] for i in sorted(set(hard+replay))])

def model_for(d):
 q,pt,pred=parent.incumbent(d);mod=importlib.import_module(q['module']);model=getattr(mod,q['class_name'])(d,.3);model.load_state_dict(pt['model']);model.full_rms_gradient=True;model.robust=None;model.linear.mask=None;return model,q,pt,pred

def inputs(data,ix,step,device):
 image=data[0][ix].to(device);clean=base.encode(image,None,100);rg=np.random.default_rng(np.random.SeedSequence([CFG['seed'],step]));theta=np.zeros((len(ix),2,3),np.float32);theta[:,0,0]=theta[:,1,1]=1;theta[:,:,2]=rg.uniform(-.04,.04,(len(ix),2));aug=base.encode(image,torch.tensor(theta,device=device),100);return clean,aug,theta

def gradients(model,x,a,y,q,opp,micro):
 model.train();model.robust=None;model.zero_grad(set_to_none=True);total=0.;prob=[]
 for off in range(0,len(y),micro):
  sl=slice(off,off+micro);pc,_=model(x[sl]);pa,_=model(a[sl]);lc=pc.clamp_min(1e-12).log();la=pa.clamp_min(1e-12).log();target=F.one_hot(y[sl],20).to(pc.dtype)*.98+.001
  ce=-.5*((target*lc).sum(1)+(target*la).sum(1)).mean();kl=(q[sl]*(q[sl].clamp_min(1e-12).log()-lc)).sum(1).mean();rows=torch.arange(len(pc),device=y.device);pair=F.softplus(lc[rows,opp[sl]]-lc[rows,y[sl]]+.3).mean();loss=ce+.5*kl+.1*pair
  (loss*len(pc)/len(y)).backward();total+=float(loss.detach())*len(pc)/len(y);prob.append(pc.detach())
 return torch.cat(prob),total

def gate(model,data,pred,plan):
 before=s.digest(model.state_dict());records=[]
 for label,ix in [('all20',np.array([np.flatnonzero(pred['train']['labels']==c)[0] for c in range(20)])),('macro16',np.array(plan['batches'][0])),('tail5',np.array(plan['batches'][1][:5]))]:
  x,a,_=inputs(data,ix,0,'cuda');y=data[1][ix].cuda();q=torch.tensor(pred['train']['probabilities'][ix],device='cuda',dtype=torch.float32);opp=torch.tensor(plan['opponent'],device='cuda')[y]
  p,_=gradients(model,x,a,y,q,opp,1);first={n:v.grad.detach().clone() for n,v in model.named_parameters()};b,_=gradients(model,x,a,y,q,opp,4);rel=[];cos=[];norms={}
  for n,v in model.named_parameters():
   aa,bb=first[n].double(),v.grad.double();norms[n]=float(bb.norm());assert torch.isfinite(bb).all() and norms[n]>0;rel.append(float((aa-bb).norm()/bb.norm()));cos.append(float((aa*bb).sum()/(aa.norm()*bb.norm())))
  assert float((p-b).abs().max())<=3e-6 and max(rel)<=.02 and min(cos)>=.999
  records.append(dict(samples=label,probability_maxabs=float((p-b).abs().max()),gradient_max_relative=max(rel),gradient_min_cosine=min(cos),gradient_norms=norms))
 assert s.digest(model.state_dict())==before
 model.eval()
 with torch.no_grad():
  p,_=model(base.encode(data[0][:4].cuda(),None,100));err=float(np.abs(p.cpu().numpy()-pred['train']['probabilities'][:4]).max());assert err<=3e-6
 return dict(passed=True,checks=records,parent_probability_maxabs=err,optimizer_steps=0,TEST_forwards=0)

def ready():
 p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();identity();parent.old.controls()

def fit(d,gpu):
 assert d in (4,6) and os.environ['CUDA_VISIBLE_DEVICES']==gpu;ready();out=OUT/f'L{d}';assert not out.exists();claim(out.with_name(out.name+'.claim.json'),'TEST_error_TRAIN_only_steps');out.mkdir();resources.slots();s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
 model,q,pt,pred=model_for(d);model=model.cuda();data=base.load('train');pl=plan(pred);s.save(out/'sampling_plan.json',pl,exclusive=True);proof=gate(model,data,pred,pl);s.save(out/'numeric_gate.json',proof,exclusive=True);ema=copy.deepcopy(model).eval();named=parent.groups(model);opt=torch.optim.AdamW([dict(params=[p for n,p in g],lr=lr,weight_decay=0) for g,lr in zip(named,[CFG['phase_lr'],CFG['electronic_lr'],CFG['alpha_lr']])]);meta=dict(identity(),config=CFG,parent=q,init_digest=s.digest(model.state_dict()),pid=os.getpid(),gpu=gpu,depth=d,new_Adam_state=True,not_fresh_weights=True,controls=parent.old.controls());s.save(out/'metadata.json',meta,exclusive=True)
 best=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());selected=0;metrics={r:engine.metrics(z['labels'],z['probabilities']) for r,z in pred.items()};trace=[];history=[];started=time.monotonic();test=base.load('test')
 engine.save_pt(out/'best.pt',dict(model=pt['model'],metrics=metrics,selected_step=0,contract=pt['contract'],sources=sources(),parent=q))
 for role,z in pred.items():base.save_np(out/('selected_'+role+'.npz'),z)
 for step,indices in enumerate(pl['batches'],1):
  assert time.monotonic()-started<1800;ix=np.array(indices);x,a,theta=inputs(data,ix,step,'cuda');y=data[1][ix].cuda();target=torch.tensor(pred['train']['probabilities'][ix],device='cuda',dtype=torch.float32);opp=torch.tensor(pl['opponent'],device='cuda')[y];p,loss=gradients(model,x,a,y,target,opp,CFG['micro'])
  if step==CFG['steps']:boundary=dict(raw_before={n:p.detach().cpu().clone() for n,p in model.named_parameters()},gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()},optimizer_before=copy.deepcopy(opt.state_dict()),ema_before={n:p.detach().cpu().clone() for n,p in ema.named_parameters()},groups=[[n for n,p in g] for g in named])
  norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));opt.step();model.project()
  with torch.no_grad():
   for ep,p in zip(ema.parameters(),model.parameters()):ep.mul_(CFG['ema']).add_(p,alpha=1-CFG['ema'])
  trace.append(dict(step=step,indices=indices,theta_sha256=s.digest(theta),loss=loss,clip_norm=norm,adam_updates=step))
  if step==CFG['steps']:boundary['clip_norm']=norm;engine.save_pt(out/'optimizer_boundary.pt',boundary)
  if step%CFG['segment']==0:
   tm,tp=base.evaluate(ema,test);base.save_np(out/f'test_step{step:03d}.npz',tp);correct=int((tp['labels']==tp['probabilities'].argmax(1)).sum());row=dict(step=step,test=tm,test_correct=correct,ema_digest=s.digest(ema.state_dict()),seconds=time.monotonic()-started)
   if correct>best:
    best=correct;selected=step;mm=dict(test=tm)
    for role in ('train','val'):
     mm[role],zz=base.evaluate(ema,data if role=='train' else base.load('val'));base.save_np(out/('selected_'+role+'.npz'),zz)
    base.save_np(out/'selected_test.npz',tp);engine.save_pt(out/'best.pt',dict(model={n:v.detach().cpu() for n,v in ema.state_dict().items()},metrics=mm,selected_step=step,contract=ema.contract(),sources=sources(),parent=q))
   history.append(row);s.save(out/'history.json',history);s.save(out/'method_trace.json',trace);s.save(out/'status.json',dict(state='short_steps_running',step=step,test=tm['accuracy'],selected_step=selected,selected_test=best/411));print(dict(depth=d,step=step,correct=correct,best=best),flush=True)
 engine.save_pt(out/'last.pt',dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},ema={n:v.detach().cpu() for n,v in ema.state_dict().items()},optimizer=opt.state_dict(),steps=CFG['steps'],sources=sources()))
 endpoint={}
 for role in ('train','val'):endpoint[role],_=base.evaluate(ema,data if role=='train' else base.load('val'))
 bp=torch.load(out/'best.pt',map_location='cpu',weights_only=False);s.save(out/'fit_result.json',dict(passed=True,actual_steps=CFG['steps'],TRAIN_presentations=CFG['steps']*16,unique_selected_TRAIN=len(pl['train_ids']),full_TRAIN_training_epochs=0,TEST_evaluations=len(history),selected_step=selected,metrics=bp['metrics'],endpoint=endpoint,seconds=time.monotonic()-started,hashes={f:s.sha(out/f) for f in ('best.pt','last.pt','history.json','method_trace.json','sampling_plan.json','optimizer_boundary.pt')}),exclusive=True);identity();parent.old.controls()

def campaign():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'two_depth_targeted_short_steps');OUT.mkdir();jobs=[];complete=[];fail=[]
 for d,g in ((4,0),(6,1)):
  gpu=resources.GPU[g];slots=resources.slots();assert slots['total']<5;s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);log=(OUT/f'L{d}.log').open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--depth',str(d),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append((d,p,log))
 while jobs:
  for d,p,log in list(jobs):
   rc=p.poll()
   if rc is None:continue
   log.close();jobs.remove((d,p,log));s.save(OUT/f'L{d}.producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True)
   if rc:fail.append(dict(depth=d,producer_returncode=rc));continue
   with (OUT/f'L{d}/audit.log').open('x') as f:arc=subprocess.run([os.sys.executable,str(HERE/FILES[2]),'--depth',str(d)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT).returncode
   if arc:fail.append(dict(depth=d,audit_returncode=arc));continue
   s.save(OUT/f'L{d}/audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(OUT/f'L{d}/science/audit.json')),exclusive=True);complete.append(d)
  s.save(OUT/'status.json',dict(state='failed' if fail else 'short_steps_running' if jobs else 'short_steps_complete',active=[dict(depth=d,pid=p.pid) for d,p,f in jobs],complete=complete,failures=fail))
  if jobs:time.sleep(2)
 assert not fail

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['fit','campaign'],required=True);a.add_argument('--depth',type=int);a.add_argument('--gpu');z=a.parse_args()
 try:campaign() if z.phase=='campaign' else fit(z.depth,z.gpu)
 except BaseException as e:
  folder=OUT/f'L{z.depth}' if z.depth else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
