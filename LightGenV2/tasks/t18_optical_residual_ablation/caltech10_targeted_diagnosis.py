"""New train-only objective/quantization diagnosis; no Adam or test forward."""
import argparse,copy,hashlib,math,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech10_sam_dropout as prior
s=prior.s;HERE=prior.HERE;ROOT=prior.ROOT
FILES=('caltech10_targeted_diagnosis.py','CALTECH10_TARGETED_DIAGNOSIS_PROTOCOL_20261015.md')
def sources():
 src=prior.sources()
 for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return src
def gradients(model,x,y,w,cfg):
 params=list(model.parameters());parts={k:[torch.zeros_like(p) for p in params] for k in ('classification','capture','smoothness')};losses={k:0. for k in parts}
 for i in range(0,len(x),8):
  p,c=model(x[i:i+8]);losses_now=dict(classification=-(y[i:i+8]*w[None]*p.clamp_min(1e-12).log()).sum(1).mean(),capture=-cfg['capture_weight']*c.clamp_min(1e-12).log().mean(),smoothness=cfg['phase_smooth_weight']*model.smoothness())
  for kind,loss in losses_now.items():
   values=torch.autograd.grad(loss,params,retain_graph=(kind=='classification'));factor=len(p)/len(x);losses[kind]+=float(loss.detach())*factor
   for total,g in zip(parts[kind],values):total.add_(g,alpha=factor)
 return {k:torch.cat([g.flatten() for g in v]) for k,v in parts.items()},losses
def phase_codes(model):
 out=[]
 for i,p in enumerate(model.parameters()):
  phase=2*math.pi*torch.sigmoid(p)
  if i:phase=torch.remainder(torch.angle((1-model.rho)*torch.exp(1j*phase)+model.rho),2*math.pi)
  out.append(torch.floor(phase/(2*math.pi)*256).clamp(0,255).to(torch.uint8).flatten())
 return torch.cat(out)
def run(a):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu==prior.resources.GPU[4] and not a.out.exists();assert prior.resources.slots()['total']==1
 root=HERE/'runs/simulation'/prior.RUN;meta=s.read(root/'metadata.json');assert meta['sources']==prior.sources() and prior.parent.base.exited(meta['pid']) and s.read(root/'status.json')['state']=='all_methods_stopped_targets_or_plateau';s.engine.verify_sources(meta['sources'],meta['git_commit']);s.gpu_budget.ALLOWED.update(prior.resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,git_commit=commit,optimizer_steps=0,test_arrays_read=0),exclusive=True);result={}
 for depth in (2,4):
  case=f'L{depth}_sam';blocks=[b for b in sorted((root/case).glob('block*')) if (b/'comparison.json').exists()];block=blocks[-1];arm=block/case;done=s.read(block/'audit_completed.json');assert done['passed'] and done['actual_audit_returncode']==0 and done['audit_sha256']==s.sha(block/'science/audit.json');last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);best=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);cfg=copy.deepcopy(best['config']);prior.parent.bind();data=prior.parent.load_fit(Path(cfg['data']),'train',prior.parent.manifest(Path(cfg['data'])));model=prior.parent.SubsetMoE(depth,.3).cuda();model.load_state_dict(best['model'],strict=True);model.eval();initial=s.digest(model.state_dict());weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).cuda();_,theta=s.engine.epoch_plan(1,len(data[1]),cfg)
  groups=[np.array([int((data[1]==k).nonzero()[0]) for k in range(10)]+[int((data[1]==k).nonzero()[1]) for k in range(6)]),np.arange(len(data[1])-16,len(data[1]))];probes=[]
  for group,idx in enumerate(groups):
   x,y,_=s.engine.mixed_batch(data,idx,theta,1,group,cfg,'cuda');g,losses=gradients(model,x,y,weights,cfg);ce=g['classification'];total=sum(g.values());norms={k:float(v.norm()) for k,v in g.items()};cos={k:float(torch.dot(ce,v)/(ce.norm()*v.norm()+1e-12)) for k,v in g.items() if k!='classification'};saved=[p.detach().clone() for p in model.parameters()];codes=phase_codes(model);offset=0;radii=[]
   for radius in (.05,.5,2.):
    offset=0
    with torch.no_grad():
     for p,w in zip(model.parameters(),saved):n=p.numel();p.copy_(w+(radius*total[offset:offset+n]/total.norm()).reshape_as(p));offset+=n
    moved=float((phase_codes(model)!=codes).float().mean());radii.append(dict(raw_vector_L2_radius=radius,changed_phase_gray_fraction=moved))
   with torch.no_grad():
    for p,w in zip(model.parameters(),saved):p.copy_(w)
   assert s.digest(model.state_dict())==initial;probes.append(dict(group=group,loss_terms=losses,gradient_norms=norms,classification_aux_cosines=cos,phase_quantum_rad=2*math.pi/256,radii=radii,exact_restore=True))
  comparison=s.read(block/'comparison.json');result[str(depth)]=dict(parent_arm=str(arm),parent_root=str(root),parent_case=case,parent_epoch=last['epoch'],parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),parent_best_sha256=s.sha(arm/'best_checkpoint.pt'),parent_history_sha256=s.sha(arm/'history.json'),parent_audit_sha256=done['audit_sha256'],parent_config=last['config'],inherited_tests=comparison['total_selected_model_tests'],best_epoch=best['epoch'],best_model_digest=initial,probes=probes)
  del model,g,total,data;torch.cuda.empty_cache()
 a.out.mkdir(parents=True);s.save(a.out/'diagnosis.json',dict(passed=True,sources=src,git_commit=commit,depths=result,optimizer_steps=0,test_images_read=0,training_groups_per_depth=2,model_parameter_state_preserved=True),exclusive=True);print('TRAIN_ONLY_DIAGNOSIS_SAVED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 try:run(a)
 except BaseException as e:
  fail=a.out.with_name(a.out.name+'.failure.json')
  if not fail.exists():s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
