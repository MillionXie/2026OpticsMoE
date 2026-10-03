"""One auxiliary visual encoder shared by ALL text/optical variants."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .prepare import COLORS, SHAPES, save, digest


class VisionEncoder(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        layers=[];cin=3
        for cout in [16,32,64]:
            layers.extend([nn.Conv2d(cin,cout,3,padding=1),nn.BatchNorm2d(cout),nn.ReLU(),nn.MaxPool2d(2)])
            cin=cout
        layers.extend([nn.Conv2d(64,128,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten()])
        self.features=nn.Sequential(*layers)
        self.head=nn.Linear(128,num_classes)

    def forward(self, x):
        z=self.features(x.permute(0,3,1,2).float()/255)
        return z,self.head(z)


def load(root, split):
    rows=json.loads((root/f'{split}_questions.json').read_text())
    images=torch.tensor(np.load(root/f'{split}_images.npz')['images'],device='cuda')
    indices=torch.tensor([r['image_local'] for r in rows],device='cuda')
    query=torch.tensor([COLORS.index(r['color'])*3+SHAPES.index(r['shape']) for r in rows],device='cuda')
    labels=torch.tensor([r['label'] for r in rows],device='cuda').float()
    return images,indices,query,labels


@torch.no_grad()
def evaluate(model,data):
    model.eval();images,indices,query,y=data
    logits=torch.cat([model(x)[1] for x in images.split(64)])
    scores=logits[indices,query]
    return dict(accuracy=float(((scores>=0)==y.bool()).float().mean()),
                nll=float(F.binary_cross_entropy_with_logits(scores,y)))


@torch.no_grad()
def frozen_features(checkpoint, images):
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model=VisionEncoder(state['model']['head.weight'].shape[0]).cuda()
    model.load_state_dict(state['model']);model.requires_grad_(False).eval()
    result=torch.cat([model(x)[0] for x in images.split(64)])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=60)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
    model=VisionEncoder().cuda();train=load(a.data,'train');val=load(a.data,'val')
    # Each image has exactly six questions (three positive, three negative).
    assert torch.equal(train[1],torch.arange(len(train[0]),device='cuda').repeat_interleave(6))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.0001)
    best=float('inf');history=[]
    save(a.out/'metadata.json',dict(seed=17,epochs=a.epochs,lr=.001,weight_decay=.0001,batch_images=32,
         purpose='Visual auxiliary diagnostic, not an optical result; head is discarded before optical training',
         git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
         data_manifest_sha256=digest((a.data/'manifest.json').read_bytes()),
         all_parameters=sum(p.numel() for p in model.parameters()),
         retained_feature_parameters=sum(p.numel() for p in model.features.parameters()),test_accessed=False))
    for epoch in range(1,a.epochs+1):
        start=time.time();model.train()
        order=torch.randperm(len(train[0]),device='cuda')
        for batch in order.split(32):
            images=train[0][batch].clone()
            flip=torch.rand(len(batch),device='cuda')<.5
            images[flip]=images[flip].flip(2)
            _,logits=model(images)
            query_rows=batch[:,None]*6+torch.arange(6,device='cuda')
            target=train[3][query_rows];query=train[2][query_rows]
            scores=logits.gather(1,query)
            cost=F.binary_cross_entropy_with_logits(scores,target)
            opt.zero_grad();cost.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        sched.step();row=dict(epoch=epoch,train=evaluate(model,train),val=evaluate(model,val),seconds=time.time()-start)
        history.append(row);save(a.out/'history.json',history)
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,metrics=row)
        torch.save(checkpoint,a.out/'last_checkpoint.pt')
        if row['val']['nll']<best:
            best=row['val']['nll'];torch.save(checkpoint,a.out/'best_checkpoint.pt')
        print(json.dumps(row),flush=True)
    save(a.out/'status.json',dict(status='complete',test_accessed=False))


if __name__=='__main__':main()
