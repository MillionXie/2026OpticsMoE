"""Cross-dataset binary Kather->EuroSAT run using the same optical graph."""
import argparse,json,platform,subprocess,sys
from pathlib import Path
import numpy as np,torch
from .model import OpticalMoE,loss
from .data import domain,sha
from .run import save,evaluate

def load_npz(path, positive):
 with np.load(path,allow_pickle=False) as z:
  out={}
  for split in ('train','validation' if 'validation_images' in z.files else 'val'):
   out[split+'_images']=z[split+'_images'].copy(); out[split+'_labels']=positive(z[split+'_labels']);out[split+'_ids']=z[split+'_ids'].copy()
 return out

def balanced(x,y,n,seed):
 r=np.random.default_rng(seed); ids=[]
 for c in (0,1): ids.extend(r.permutation(np.flatnonzero(y==c))[:n])
 return np.array(ids)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--kather',type=Path,required=True);p.add_argument('--eurosat',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
 k=load_npz(a.kather,lambda y:(y!=0).astype(np.int64)); w=load_npz(a.eurosat,lambda y:np.isin(y,[1,2,5,8,9]).astype(np.int64));
 # equalized subsets keep the cross-dataset comparison practical and class-balanced
 ka=balanced(k['train_images'],k['train_labels'],min(600,min(np.bincount(k['train_labels']))),cfg['seed']); wb=balanced(w['train_images'],w['train_labels'],min(600,min(np.bincount(w['train_labels']))),cfg['seed']+1)
 xA=torch.from_numpy(k['train_images'][ka]); yA=torch.from_numpy(k['train_labels'][ka]).long(); xB=torch.from_numpy(w['train_images'][wb]);yB=torch.from_numpy(w['train_labels'][wb]).long();vxA=torch.from_numpy(k['val_images']);vyA=torch.from_numpy(k['val_labels']).long();vxB=torch.from_numpy(w['val_images']);vyB=torch.from_numpy(w['val_labels']).long()
 save(a.out/'split.json',dict(dataset_A='Kather tumor vs non-tumor',dataset_B='EuroSAT natural vs artificial/agricultural',A_counts=np.bincount(yA.numpy(),minlength=2).tolist(),B_counts=np.bincount(yB.numpy(),minlength=2).tolist(),validation_A=np.bincount(vyA.numpy(),minlength=2).tolist(),validation_B=np.bincount(vyB.numpy(),minlength=2).tolist(),test_images_read=False))
 save(a.out/'metadata.json',dict(command=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.version,torch=torch.__version__,device=a.device,scope='cross-dataset binary preliminary',kather_source=str(a.kather),eurosat_source=str(a.eurosat),test_images_read=False))
 m=OpticalMoE(cfg).to(a.device); rng=np.random.default_rng(cfg['seed']); hist=[]; before=None
 def ev(x,y): return evaluate(m,x,y,'A',cfg['batch_size'])[0]
 for stage in ('A','warmup','B'):
  m.configure(stage); frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}; groups=[dict(params=[p for p in m.experts if p.requires_grad],lr=cfg['lr_expert'])]
  if stage!='warmup':groups.append(dict(params=[m.router,m.global_phase],lr=cfg['lr_shared']))
  opt=torch.optim.Adam(groups); X,Y=(xA,yA) if stage=='A' else (xB,yB); epochs=cfg['epochs_'+stage]
  for ep in range(epochs):
   order=rng.permutation(len(Y)); total=0
   for st in range(0,len(Y),cfg['batch_size']):
    ids=order[st:st+cfg['batch_size']]; xb=X[ids].to(a.device);yb=Y[ids].to(a.device);opt.zero_grad(set_to_none=True);o=m(xb,warmup=stage=='warmup');v=loss(o,yb);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=v.item()*len(yb)
   ma=ev(vxA,vyA); m.eval();
   with torch.no_grad():
    pb=[]
    for st in range(0,len(vyB),cfg['batch_size']):pb.append(m(vxB[st:st+cfg['batch_size']].to(a.device))['probabilities'].cpu())
    pb=torch.cat(pb);mb={'accuracy':float((pb.argmax(1)==vyB).float().mean())}
   hist.append(dict(stage=stage,epoch=ep+1,train_loss=total/len(Y),val_A=ma['accuracy'],val_B=mb['accuracy']));print(json.dumps(hist[-1]),flush=True)
  for n,p in m.named_parameters():
   if n in frozen and not torch.equal(p,frozen[n]):raise RuntimeError('Frozen changed '+n)
  if stage!='warmup':torch.save(dict(model=m.state_dict(),config=cfg,stage=stage,history=hist),a.out/stage+'.pt')
  if stage=='A':before=ev(vxA,vyA)
 save(a.out/'history.json',hist);finalA=ev(vxA,vyA);m.eval();
 with torch.no_grad():
  outs=[]
  for st in range(0,len(vyB),cfg['batch_size']):outs.append(m(vxB[st:st+cfg['batch_size']].to(a.device))['probabilities'].cpu())
 finalB={'accuracy':float((torch.cat(outs).argmax(1)==vyB).float().mean())}
 save(a.out/'metrics.json',dict(A_before=before,A_after=finalA,B_after=finalB,BWT=finalA['accuracy']-before['accuracy'],dataset_A='Kather tumor/non-tumor',dataset_B='EuroSAT natural/artificial'))
if __name__=='__main__':main()



