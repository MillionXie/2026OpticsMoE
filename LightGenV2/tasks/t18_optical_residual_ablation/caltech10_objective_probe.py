"""Fixed TRAIN gradient trade-off and photometric robustness probe; no Adam."""
import argparse,copy,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech10_consistency_guard_repair as repair
prior=repair.failed.prior;s=prior.s;HERE=prior.HERE;ROOT=prior.ROOT
FILES=('caltech10_objective_probe.py','CALTECH10_OBJECTIVE_PROBE_PROTOCOL_20261015.md')
OUT=HERE/'runs/smoke/caltech10_objective_photometric_probe_20261015'
def sources():
 src=repair.sources()
 for name in FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
 return src
def gradients(model,teacher,x,y,weights,cfg):
 params=list(model.parameters());parts={name:[torch.zeros_like(q) for q in params] for name in ('classification','distill_weighted','focal15','capture','smoothness')};losses={name:0. for name in parts};active=0;correct=0
 for start in range(0,len(x),8):
  p,c=model(x[start:start+8]);target_y=y[start:start+8];factor=len(p)/len(x)
  with torch.no_grad():
   tp,_=teacher(x[start:start+8]);gate=(tp.argmax(1)==target_y.argmax(1))&(tp.max(1).values>=.5);soft=torch.softmax(tp.clamp_min(1e-12).log()/2.,1)
  per_ce=-(target_y*weights[None]*p.clamp_min(1e-12).log()).sum(1);kl=(soft*(soft.clamp_min(1e-12).log()-torch.log_softmax(p.clamp_min(1e-12).log()/2.,1))).sum(1)*4
  terms=dict(classification=per_ce.mean(),distill_weighted=.5*(kl*gate).mean(),focal15=((1-(target_y*p).sum(1)).clamp_min(0).pow(1.5)*per_ce).mean(),capture=-cfg['capture_weight']*c.clamp_min(1e-12).log().mean(),smoothness=cfg['phase_smooth_weight']*model.smoothness())
  for name,loss in terms.items():
   assert torch.isfinite(loss);g=torch.autograd.grad(loss,params,retain_graph=True);losses[name]+=float(loss.detach())*factor
   for acc,v in zip(parts[name],g):assert torch.isfinite(v).all();acc.add_(v,alpha=factor)
  active+=int(gate.sum());correct+=int((p.argmax(1)==target_y.argmax(1)).sum())
 vectors={name:torch.cat([q.flatten() for q in gs]) for name,gs in parts.items()};norms={name:float(g.norm()) for name,g in vectors.items()};ce=vectors['classification'];cos={name:float(torch.dot(ce,g)/(ce.norm()*g.norm()+1e-12)) for name,g in vectors.items() if name!='classification'}
 return dict(loss_terms=losses,gradient_norms=norms,classification_cosines=cos,teacher_gate_fraction=active/len(x),student_accuracy=correct/len(x),examples=len(x))
def summary(p,y):
 pred=p.argmax(1);return dict(accuracy=float((pred==y).double().mean()),per_class_recall=[float((pred[y==k]==k).double().mean()) for k in range(10)],mean_confidence=float(p.max(1).values.mean()))
def run(a):
 assert a.out==OUT and not OUT.exists() and not OUT.with_name(OUT.name+'.claim.json').exists() and not OUT.with_name(OUT.name+'.failure.json').exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu==prior.resources.GPU[4];repair.exclusive_self(prior.resources.slots()['jobs'],os.getpid(),a.gpu);repair.preserved_failure();proof=s.read(repair.OUT/'diagnosis.json');assert proof['passed'] and proof['sources']==repair.sources() and s.sha(repair.OUT/'diagnosis.json')=='7ce258a2d3b94caaf4ca5e56e3b66a1f545dedac0e7afc37a316df573c4b8661';s.engine.verify_sources(proof['sources'],proof['git_commit']);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);assert len(src)==209;s.save(OUT.with_name(OUT.name+'.claim.json'),dict(sources=src,git_commit=commit,optimizer_steps=0,test_arrays_read=0),exclusive=True)
 s.gpu_budget.ALLOWED.update(prior.resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);prior.parent.bind();parents=proof['parents'];arm=Path(parents['L2_distill']['parent_arm']);assert s.sha(arm/'best_checkpoint.pt')==parents['L2_distill']['parent_best_sha256'];ck=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);cfg=copy.deepcopy(ck['config']);data=prior.parent.load_fit(Path(cfg['data']),'train',prior.parent.manifest(Path(cfg['data'])));student=prior.parent.SubsetMoE(2,.3).cuda();student.load_state_dict(ck['model'],strict=True);student.eval();initial=s.digest(student.state_dict());teacher=prior.teacher_model(student.raw_router.device);weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).cuda();_,theta=s.engine.epoch_plan(1041,len(data[1]),cfg)
 groups=dict(balanced20=np.array([int((data[1]==k).nonzero()[j]) for k in range(10) for j in range(2)]),hard_bonsai_airplanes20=np.array([int((data[1]==k).nonzero()[j]) for k in (4,3) for j in range(10)]));results={}
 for batch,(name,idx) in enumerate(groups.items()):x,y,_=s.engine.mixed_batch(data,idx,theta,1041,batch,cfg,'cuda');results[name]=gradients(student,teacher,x,y,weights,cfg)
 photometric={}
 with torch.no_grad():
  for name,gains,contrast in [('neutral',[1.,1.,1.],1.),('warm_contrast',[1.2,1.,.8],1.2),('cool_lowcontrast',[.8,1.,1.2],.8)]:
   probs=[]
   for idx in torch.arange(len(data[1])).split(8):
    images=data[0][idx].cuda().float()/255.;mean=images.mean(dim=(-2,-1),keepdim=True);images=((images-mean)*contrast+mean)*torch.tensor(gains,device='cuda')[None,:,None,None];images=(images.clamp(0,1)*255).round().to(torch.uint8);p,_=teacher(s.engine.input_amplitude(images));probs.append(p.cpu())
   photometric[name]=summary(torch.cat(probs),data[1])
 assert s.digest(student.state_dict())==initial and s.digest(teacher.state_dict())==prior.P['teacher_model_digest'] and all(q.grad is None for q in student.parameters()) and all(q.grad is None and not q.requires_grad for q in teacher.parameters());repair.preserved_failure();OUT.mkdir(parents=True);s.save(OUT/'diagnosis.json',dict(passed=True,sources=src,git_commit=commit,parents=parents,prior_diagnosis_sha256=s.sha(repair.OUT/'diagnosis.json'),gradient_groups=results,teacher_photometric_training=photometric,classes=cfg['classes'],focal_gamma=1.5,optimizer_steps=0,test_arrays_read=0,validation_arrays_read=0,model_state_preserved=True),exclusive=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 try:run(a)
 except BaseException as exc:
  failure=OUT.with_name(OUT.name+'.failure.json')
  if not failure.exists():s.save(failure,dict(error=repr(exc),automatic_retry=False),exclusive=True)
  raise
