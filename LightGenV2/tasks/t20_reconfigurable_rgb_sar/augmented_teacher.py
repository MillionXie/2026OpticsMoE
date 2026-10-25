"""TRAIN-only offline image teacher; never part of optical inference.

The teacher is deliberately larger than the fixed optical frontend. Its capacity
and both-domain supervision are reported, not attributed to the student graph.
"""
import argparse,json,time,math,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .train_unified_static import load_arrays,balanced_batches
from .refine_unified import scores_from_logits
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,gpu_gate


def block(a,b,pool=False):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU(),
                         nn.MaxPool2d(2) if pool else nn.Identity())


class ImageTeacher(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Sequential(block(3,32),block(32,64,True))
        self.res1=nn.Sequential(block(64,64),block(64,64))
        self.down=nn.Sequential(block(64,128,True),block(128,256,True))
        self.res2=nn.Sequential(block(256,256),block(256,256))
        self.head=nn.Linear(256,10)
        self.register_buffer('mean',torch.zeros(1,3,1,1));self.register_buffer('std',torch.ones(1,3,1,1))

    def forward(self,x):
        x=(x.float()/255.-self.mean)/self.std;x=self.stem(x);x=x+self.res1(x)
        x=self.down(x);x=x+self.res2(x);return self.head(x.mean((2,3)))


def tensor_images(images,device):
    return torch.as_tensor(np.ascontiguousarray(images.transpose(0,3,1,2)),device=device)


def augment(x,generator):
    """Label-preserving TRAIN D4 and small reflected translations; no mix labels."""
    k=int(torch.randint(0,4,(),generator=generator));x=torch.rot90(x,k,(-2,-1))
    if bool(torch.rand((),generator=generator)<.5):x=x.flip(-1)
    dy,dx=torch.randint(0,5,(2,),generator=generator).tolist()
    x=F.pad(x.float(),(2,2,2,2),mode='reflect')[:,:,dy:dy+56,dx:dx+56]
    return x


@torch.no_grad()
def teacher_logits(model,data,device,batch=128):
    model.eval();return torch.cat([model(tensor_images(data['images'][i:i+batch],device)).cpu()
                                  for i in range(0,len(data['labels']),batch)])


def teacher_score(logits,data):
    return scores_from_logits(logits.numpy(),data)['scores']


def load_teacher(run,data,device):
    receipt=json.loads((run/'completion.json').read_text());path=run/'best_checkpoint.pt'
    assert receipt['TEST_access'] is False and sha(path)==receipt['best_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['teacher_only'] and cp['TEST_access'] is False
    assert cp['TRAIN_identity_SHA256']==sha(data/'source_data/data/data.npz')
    model=ImageTeacher().to(device);model.load_state_dict(cp['model'],strict=True)
    return model.eval().requires_grad_(False),receipt


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('data','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    cfg=json.loads(a.profile.read_text());assert cfg==json.loads(Path(__file__).with_name('config_augmented_teacher_20261025.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0')
    tr=load_arrays(a.data,'train');va=load_arrays(a.data,'validation');m=ImageTeacher().to(device)
    pixels=torch.from_numpy(tr['images'].transpose(0,3,1,2)).float()/255.
    with torch.no_grad():m.mean.copy_(pixels.mean((0,2,3),keepdim=True).to(device));m.std.copy_(pixels.std((0,2,3),keepdim=True).clamp_min(.05).to(device))
    del pixels
    ix=next(balanced_batches(tr,[0,1],cfg['batch_size'],42));cost=F.cross_entropy(m(tensor_images(tr['images'][ix],device)),torch.as_tensor(tr['labels'][ix],device=device));cost.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if a.preflight_only:print(json.dumps(dict(passed=True,loss=float(cost.detach()),parameters=sum(p.numel() for p in m.parameters()),TEST_loaded=False)));return
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();history=[];best=(-1.,);best_epoch=0;last_improvement=0;source=sha(Path(__file__))
    protocol=dict(config=cfg,teacher_only=True,teacher_parameters=sum(p.numel() for p in m.parameters()),TRAIN_identity_SHA256=sha(a.data/'source_data/data/data.npz'),TRAIN_IDs=tr['ids'].tolist(),dataset_manifest_SHA256=sha(a.data/'manifest.json'),source_SHA256=source,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),TRAIN=6000,validation=2000,TEST_access=False,SDK=False,gpu=gpu)
    put(a.out/'protocol.json',protocol);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);rng=torch.Generator().manual_seed(cfg['seed'])
    def save(role,epoch):torch.save(dict(model={n:v.detach().cpu() for n,v in m.state_dict().items()},epoch=epoch,teacher_only=True,TEST_access=False,TRAIN_identity_SHA256=protocol['TRAIN_identity_SHA256']),a.out/(role+'_checkpoint.pt'))
    for epoch in range(1,cfg['epochs']+1):
        m.train();total=0.;correct=0;seen=0
        for g in opt.param_groups:g['lr']=cfg['lr']*(.03+.97*.5*(1+math.cos(math.pi*(epoch-1)/cfg['epochs'])))
        for ix in balanced_batches(tr,[0,1],cfg['batch_size'],cfg['seed']+epoch):
            x=augment(tensor_images(tr['images'][ix],device),rng);y=torch.as_tensor(tr['labels'][ix],device=device);d=torch.as_tensor(tr['domains'][ix],device=device);logits=m(x)
            costs=F.cross_entropy(logits,y,reduction='none',label_smoothing=.02);w=torch.where(d==0,2.,1.);cost=(costs*w).sum()/w.sum()
            opt.zero_grad(set_to_none=True);cost.backward();nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
            total+=float(cost.detach())*len(ix);correct+=int((logits.argmax(1)==y).sum());seen+=len(ix)
        scores=teacher_score(teacher_logits(m,va,device),va);key=((2*scores['RGB']['accuracy']+scores['SAR']['accuracy'])/3.,min(s['accuracy'] for s in scores.values()))
        if key>best:best=key;best_epoch=epoch;last_improvement=epoch;save('best',epoch);put(a.out/'selected_validation.json',scores)
        save('last',epoch);row=dict(epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy=correct/seen,validation=scores,best_epoch=best_epoch);history.append(row);put(a.out/'history.json',history);put(a.out/'status.json',dict(state='training',**row,updated=time.time()))
        assert sha(Path(__file__))==source
        if time.monotonic()-start>cfg['max_seconds'] or (epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']):break
    cp=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);m.load_state_dict(cp['model'],strict=True);val=teacher_score(teacher_logits(m,va,device),va)
    put(a.out/'completion.json',dict(passed=True,teacher_only=True,TEST_access=False,best_epoch=best_epoch,epochs_completed=epoch,validation=val,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),seconds=time.monotonic()-start,SDK=False));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
