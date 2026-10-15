"""Fixed TRAIN-only clean/affine robustness and teacher coverage diagnosis."""
import argparse,copy,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech10_targeted_refinement as prior
s=prior.s;HERE=prior.HERE;ROOT=prior.ROOT
FILES=('caltech10_consistency_diagnosis.py','CALTECH10_CONSISTENCY_DIAGNOSIS_PROTOCOL_20261015.md')
def sources():
 out=prior.sources()
 for name in FILES:out[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
 return out
def summary(p,y):
 pred=p.argmax(1);gate=(pred==y)&(p.max(1).values>=.5)
 return dict(accuracy=float((pred==y).double().mean()),gate_fraction=float(gate.double().mean()),per_class_recall=[float((pred[y==k]==k).double().mean()) for k in range(10)],per_class_gate=[float(gate[y==k].double().mean()) for k in range(10)],mean_confidence=float(p.max(1).values.mean()))
def run(a):
 assert a.out==HERE/'runs/smoke/caltech10_consistency_train_diagnosis_20261015' and not a.out.exists() and not a.out.with_name(a.out.name+'.claim.json').exists()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu==prior.resources.GPU[4] and prior.resources.slots()['total']==0
 root=HERE/'runs/simulation'/prior.RUN;meta=s.read(root/'metadata.json');assert prior.parent.base.exited(meta['pid']) and s.read(root/'status.json')['state']=='all_targeted_stopped_targets_or_review' and meta['sources']==prior.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
 prior.parent.bind();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,git_commit=commit,training_only=True,optimizer_steps=0,test_arrays_read=0),exclusive=True)
 s.gpu_budget.ALLOWED.update(prior.resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);models={};parents={};probabilities={};method_coverage={}
 for case in prior.CASES:
  blocks=[b for b in sorted((root/case).glob('block*')) if (b/'comparison.json').exists()];b=blocks[-1];arm=b/case;rec=s.read(arm/'producer_exit.json');done=s.read(b/'audit_completed.json');assert rec['actual_exit'] and rec['returncode']==0 and prior.parent.base.exited(rec['pid']) and done['passed'] and done['actual_audit_returncode']==0 and done['audit_sha256']==s.sha(b/'science/audit.json');best=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);cfg=copy.deepcopy(last['config']);m=prior.parent.SubsetMoE(prior.CASES[case]['depth'],.3).cuda();m.load_state_dict(best['model'],strict=True);m.eval();models[case]=m
  parents[case]=dict(parent_arm=str(arm),parent_root=str(root),parent_case=case,parent_epoch=last['epoch'],parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),parent_best_sha256=s.sha(arm/'best_checkpoint.pt'),parent_history_sha256=s.sha(arm/'history.json'),parent_audit_sha256=done['audit_sha256'],parent_config=last['config'],best_epoch=best['epoch'],best_model_digest=s.digest(best['model']))
  if case=='L2_distill':
   records=[v for p in sorted(arm.glob('method_epoch*.json')) for v in s.read(p)['records']];method_coverage=dict(total_samples=sum(v['shape'][0] for v in records),active_samples=sum(v['teacher_active_samples'] for v in records),macro_batches=len(records),mean_recorded_KL=float(np.mean([v['teacher_kl'] for v in records])))
 data=prior.parent.load_fit(Path(cfg['data']),'train',prior.parent.manifest(Path(cfg['data'])));labels=data[1];n=len(labels)
 for view in ('clean','affine1','affine2'):
  if view!='clean':_,theta=s.engine.epoch_plan(int(view[-1]),n,cfg)
  probabilities[view]={case:[] for case in models}
  with torch.no_grad():
   for idx in torch.arange(n).split(8):
    images=data[0][idx].cuda();transform=None if view=='clean' else torch.as_tensor(theta[idx.numpy()],device='cuda');x=s.engine.input_amplitude(images) if transform is None else s.engine.input_amplitude(images,transform)
    for case,m in models.items():p,_=m(x);probabilities[view][case].append(p.cpu())
  for case in models:probabilities[view][case]=torch.cat(probabilities[view][case]);assert torch.isfinite(probabilities[view][case]).all()
 result={view:{case:summary(p,labels) for case,p in cases.items()} for view,cases in probabilities.items()};teacher=probabilities['clean']['L4_qsam'];temperature=2.;target=torch.softmax(teacher.clamp_min(1e-12).log()/temperature,1);consistency={}
 for view in ('affine1','affine2'):
  p=probabilities[view]['L4_qsam'];kl=(target*(target.clamp_min(1e-12).log()-torch.log_softmax(p.clamp_min(1e-12).log()/temperature,1))).sum(1)*temperature**2
  consistency[view]=dict(prediction_flip_fraction=float((p.argmax(1)!=teacher.argmax(1)).double().mean()),mean_KL_clean_teacher_to_augmented=float(kl.mean()),per_class_KL=[float(kl[labels==k].mean()) for k in range(10)])
 for case,m in models.items():assert s.digest(m.state_dict())==parents[case]['best_model_digest'] and all(q.grad is None for q in m.parameters())
 a.out.mkdir(parents=True);s.save(a.out/'diagnosis.json',dict(passed=True,sources=src,git_commit=commit,parents=parents,classes=cfg['classes'],training_support=torch.bincount(labels,minlength=10).tolist(),views=result,teacher_consistency=consistency,actual_KD_coverage=method_coverage,model_state_preserved=True,optimizer_steps=0,test_arrays_read=0,validation_arrays_read=0,real_training_examples=n,views_per_model=3),exclusive=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 try:run(a)
 except BaseException as exc:
  failure=a.out.with_name(a.out.name+'.failure.json')
  if not failure.exists():s.save(failure,dict(error=repr(exc),automatic_retry=False),exclusive=True)
  raise
