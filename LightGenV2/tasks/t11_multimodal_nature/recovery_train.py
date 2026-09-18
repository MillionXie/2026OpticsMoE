"""Matched-budget optical recovery training, automatic validation selection and test."""
import argparse, hashlib, json, os, random, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score, recall_score, confusion_matrix
from LightGenV2.tasks.t09_multimodal_matching.model import OpticalOEO

def save(path,x):
 tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(x,indent=2)+'\n'); tmp.replace(path)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(y,p,rows,task,thresholds=None):
 if task=='sen12ms':
  pred=p.argmax(1); return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=list(range(10)),average='macro',zero_division=0)),balanced_accuracy=float(recall_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),class_support=np.bincount(y,minlength=10).tolist(),confusion_matrix=confusion_matrix(y,pred,labels=list(range(10))).tolist())
 events=sorted({r['event'] for r in rows}); per={}; scores=[]; aucs=[]; f1s=[]; pred=np.zeros(len(y),int)
 for ev in events:
  ix=np.array([r['event']==ev for r in rows]); yy=y[ix]; ss=p[ix,1]; threshold=(thresholds or {}).get(ev,.5); pp=ss>=threshold; pred[ix]=pp
  ap=float(average_precision_score(yy,ss)) if (yy==1).any() else None
  auc=float(roc_auc_score(yy,ss)) if len(np.unique(yy))==2 else None
  f=float(f1_score(yy,pp,zero_division=0)); f1s.append(f)
  if ap is not None:scores.append(ap)
  if auc is not None:aucs.append(auc)
  per[ev]=dict(n=len(yy),positive=int(yy.sum()),ap=ap,auroc=auc,f1=f,threshold=threshold)
 return dict(accuracy=float(accuracy_score(y,pred)),macro_ap=float(np.mean(scores)) if scores else None,macro_ap_supported_classes=len(scores),macro_auroc=float(np.mean(aucs)) if aucs else None,macro_f1=float(np.mean(f1s)),micro_ap=float(average_precision_score(y,p[:,1])),balanced_accuracy=float(recall_score(y,pred,average='macro',zero_division=0)),per_event=per)

def thresholds(y,p,rows):
 out={}
 for ev in sorted({r['event'] for r in rows}):
  ix=np.array([r['event']==ev for r in rows]); yy=y[ix]; ss=p[ix,1]
  if len(np.unique(yy))<2:out[ev]=.5;continue
  candidates=np.linspace(.01,.99,99); fs=[f1_score(yy,ss>=t,zero_division=0) for t in candidates]
  best=max(fs); out[ev]=float(min([t for t,f in zip(candidates,fs) if f==best],key=lambda t:abs(t-.5)))
 return out

def load(root,split):
 d=np.load(root/(split+'.npz')); return torch.from_numpy(d['fields']),d['labels'],json.loads((root/(split+'_records.json')).read_text())

@torch.no_grad()
def evaluate(model,data,batch,task,th=None):
 fields,y,rows=data; model.eval(); probs=[]; qs=[]; energies=[]
 for i in range(0,len(y),batch):
  out=model(fields[i:i+batch].to('cuda',dtype=torch.float32)); p=out['probabilities'];p=p/p.sum(1,keepdim=True);probs.append(p.cpu().numpy());energies.append(out['capture'].cpu().numpy())
  if model.router_phase is not None:qs.append(out['route_power'].cpu().numpy())
 p=np.concatenate(probs); e=np.concatenate(energies); result=metrics(y,p,rows,task,th);result['nll']=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean());result['zero_readout_fraction']=float((e<=1e-12).mean());q=np.concatenate(qs) if qs else None;result['route_mean']=q.mean(0).tolist() if q is not None else None
 return result,p,q,e

def augment(x,task,seed):
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(seed);return _augment(x,task)

def _augment(x,task):
 # Apply exactly the same transforms and random draws to the two architectures.
 x=x.clone()
 if task=='sen12ms':
  h=torch.rand((len(x),1,1),device=x.device)<.5;v=torch.rand((len(x),1,1),device=x.device)<.5
  boxes=[(0,112,0,112),(112,224,0,112),(0,112,112,168),(0,112,168,224),(112,224,112,168),(112,224,168,224)]
  for y0,y1,x0,x1 in boxes:
   t=x[:,y0:y1,x0:x1];t=torch.where(h,t.flip(-1),t);t=torch.where(v,t.flip(-2),t);x[:,y0:y1,x0:x1]=t
 else:
  # Time/frequency masks only in the audio half, never the text grid.
  for i in range(len(x)):
   y0=int(torch.randint(0,209,(),device=x.device));x0=int(torch.randint(0,105,(),device=x.device));x[i,y0:y0+16,:112]=0;x[i,:,x0:x0+8]=0
  left=x[:,:,:112];x[:,:,:112]=left*(.5/left.square().sum((-2,-1),keepdim=True).clamp_min(1e-20)).sqrt()
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arch',choices=['moe','d2nn'],required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=17);p.add_argument('--lr',type=float,default=.01);p.add_argument('--phase-dropout',type=float,default=.05);p.add_argument('--balance',type=float,default=.02);p.add_argument('--weight-power',type=float,default=1.);p.add_argument('--augment',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 try:
  torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
  manifest=json.loads((a.data/'manifest.json').read_text())
  for name,h in manifest.items():assert sha(a.data/name)==h,name
  protocol=json.loads((a.data/'protocol.json').read_text());task=protocol['task'];classes=protocol['classes'];train=load(a.data,'train');val=load(a.data,'val')
  model=OpticalOEO(a.arch,a.seed,phase_dropout=a.phase_dropout,input_layout='left_right',oeo_activation='centered_leaky_relu').cuda()
  if classes==10:model.class_centers=[(y,x) for y in (160,358) for x in (80,170,259,348,438)]
  assert (tuple(model.first_phase.shape)==(478,478) and model.router_phase is None) if a.arch=='d2nn' else tuple(model.first_phase.shape)==(4,224,224)
  meta=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},protocol=protocol,manifest_sha=sha(a.data/'manifest.json'),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model_source_sha=sha(Path(__file__).resolve().parents[1]/'t09_multimodal_matching/model.py'),trainer_sha=sha(Path(__file__)),parameters=sum(x.numel() for x in model.parameters()),d2nn='one full field bilinear 224->478, one aperture; no router',oeo='each layer intensity / mean, nonaffine LN, leaky_relu 0.1, softsign, unit L2 signed amplitude',selection='validation macro AP' if task=='sonyc' else 'validation macro F1',regularization=dict(phase_dropout=a.phase_dropout,moe_balance=a.balance,augmentation=a.augment,weight_power=a.weight_power))
  save(a.out/'metadata.json',meta);save(a.out/'status.json',dict(status='running',pid=os.getpid()))
  fields,y,rows=train; weights=np.ones(len(y),np.float32)
  if task=='sonyc':
   for ev in sorted({r['event'] for r in rows}):
    mask=np.array([r['event']==ev for r in rows])
    for label in [0,1]:
     ix=mask&(y==label)
     if ix.any():weights[ix]=1/int(ix.sum())
   weights=weights**a.weight_power;weights/=weights.mean()
  optimizer=torch.optim.Adam(model.parameters(),lr=a.lr);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=a.lr*.1);best=-float('inf');bestnll=float('inf');history=[]
  for epoch in range(1,a.epochs+1):
   started=time.time();model.train();order=np.random.default_rng(a.seed+epoch).permutation(len(y));grad_norm=0.;total=0.
   for i in range(0,len(y),a.batch):
    ix=order[i:i+a.batch];x=fields[ix].to('cuda',dtype=torch.float32);x=augment(x,task,a.seed+epoch*100000+i) if a.augment else x;target=torch.tensor(y[ix],device='cuda');w=torch.tensor(weights[ix],device='cuda');optimizer.zero_grad(set_to_none=True);out=model(x);prob=out['probabilities'];prob=prob/prob.sum(1,keepdim=True);cost=(F.nll_loss(prob.clamp_min(1e-12).log(),target,reduction='none')*w).mean()
    if a.arch=='moe':cost=cost+a.balance*(out['route_power'].mean(0)-.25).square().sum()
    assert torch.isfinite(cost);cost.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);grad_norm=max(grad_norm,float(gn));optimizer.step();total+=float(cost.detach())*len(ix)
   scheduler.step();tr,_,_,_=evaluate(model,train,a.batch,task);va,_,_,_=evaluate(model,val,a.batch,task);score=va['macro_ap' if task=='sonyc' else 'macro_f1'];row=dict(epoch=epoch,train=tr,val=va,loss=total/len(y),gradient_norm_max=grad_norm,seconds=time.time()-started);history.append(row);save(a.out/'history.json',history)
   checkpoint=dict(model=model.state_dict(),epoch=epoch,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),meta=meta,train=tr,val=va);torch.save(checkpoint,a.out/'last.pt')
   if score>best or (score==best and va['nll']<bestnll):best=score;bestnll=va['nll'];torch.save(checkpoint,a.out/'best.pt')
   save(a.out/'status.json',dict(status='training',epoch=epoch,pid=os.getpid(),best_validation=best));print(json.dumps(dict(epoch=epoch,arch=a.arch,task=task,train_accuracy=tr['accuracy'],val_accuracy=va['accuracy'],val_selection=score,route=va['route_mean'],seconds=row['seconds'])),flush=True)
  cp=torch.load(a.out/'best.pt',weights_only=False);model.load_state_dict(cp['model']);va,vp,vq,ve=evaluate(model,val,a.batch,task);th=thresholds(val[1],vp,val[2]) if task=='sonyc' else None;save(a.out/'thresholds.json',th)
  result=dict(selected_epoch=cp['epoch'],checkpoint_sha=sha(a.out/'best.pt'),threshold_source='validation only',metrics={})
  for split,data in [('train',train),('val',val),('test',load(a.data,'test'))]:
   raw,pred,q,e=evaluate(model,data,a.batch,task);cal=metrics(data[1],pred,data[2],task,th);result['metrics'][split]=dict(default=raw,calibrated=cal);np.savez_compressed(a.out/(split+'_predictions.npz'),probabilities=pred,labels=data[1],route=q if q is not None else np.empty((len(pred),0)),capture=e)
  save(a.out/'results.json',result);save(a.out/'status.json',dict(status='complete',selected_epoch=cp['epoch']));print('COMPLETE '+json.dumps(result['metrics']['test']),flush=True)
 except Exception:
  save(a.out/'status.json',dict(status='failed',error=traceback.format_exc()));raise
if __name__=='__main__':main()
