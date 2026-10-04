"""Train one small audio CNN shared and frozen for both optical models."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .vision import VisionEncoder
from .prepare import save,digest


@torch.no_grad()
def evaluate(model,images,labels):
    model.eval();logits=torch.cat([model(x)[1] for x in images.split(64)])
    return dict(accuracy=float((logits.argmax(1)==labels).float().mean()),nll=float(F.cross_entropy(logits,labels)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    data={};manifest=json.loads((a.data/'manifest.json').read_text())
    for split in ['train','val']:
        f=a.data/f'{split}_images.npz';assert digest(f.read_bytes())==manifest['files'][f.name]
        images=torch.tensor(np.load(f)['images'],device='cuda')
        q=a.data/f'{split}_questions.json';assert digest(q.read_bytes())==manifest['files'][q.name]
        rows=json.loads(q.read_text());assert len(rows)==2*len(images)
        labels=torch.tensor([rows[i*2]['audio_class'] for i in range(len(images))],device='cuda')
        assert all(r['image_local']==i//2 for i,r in enumerate(rows))
        data[split]=(images,labels)
    model=VisionEncoder(8).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.0001)
    save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seed=17,epochs=a.epochs,lr=.001,batch=64,weight_decay=.0001,
        data_manifest_sha256=digest((a.data/'manifest.json').read_bytes()),
        retained_parameters=sum(p.numel() for p in model.features.parameters()),
        temporary_head_parameters=sum(p.numel() for p in model.head.parameters()),
        objective='8-class audio auxiliary CE; head removed before optical matching',test_accessed=False))
    save(a.out/'status.json',dict(status='running'));best=float('inf');history=[]
    for epoch in range(1,a.epochs+1):
        model.train();images,labels=data['train']
        for idx in torch.randperm(len(images),device='cuda').split(64):
            _,logits=model(images[idx]);loss=F.cross_entropy(logits,labels[idx])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        sched.step();row=dict(epoch=epoch,train=evaluate(model,*data['train']),val=evaluate(model,*data['val']))
        history.append(row);save(a.out/'history.json',history)
        ckpt=dict(model=model.state_dict(),epoch=epoch,metrics=row)
        torch.save(ckpt,a.out/'last_checkpoint.pt')
        if row['val']['nll']<best:best=row['val']['nll'];torch.save(ckpt,a.out/'best_checkpoint.pt')
        print(json.dumps(row),flush=True)
    save(a.out/'status.json',dict(status='complete',test_accessed=False))


if __name__=='__main__':main()
