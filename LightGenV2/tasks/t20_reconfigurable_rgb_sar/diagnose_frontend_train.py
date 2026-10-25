"""Auxiliary domain-pretraining diagnosis, NOT optical performance/deployment.

Keeps the existing32128-feature/3096-temporary24head architecture. Uses only
audited TRAIN/validation. Original frontend checkpoint and T21 are never edited.
"""
import argparse,json,time,subprocess,math
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
from .unified_native import FRONTEND_SHA256
from .train_unified_static import load_arrays,balanced_batches,selection
from .refine_unified import scores_from_logits
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,gpu_gate


@torch.no_grad()
def score(model,x,data):
    model.eval();logits=torch.cat([model(v)[1][:,:10].cpu() for v in x.split(128)])
    return scores_from_logits(logits,data)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('data','frontend','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();cfg=json.loads(a.profile.read_text())
    if cfg!=json.loads(Path(__file__).with_name('config_frontend_diagnostic_20261025.json').read_text()):raise ValueError('Exact finite diagnostic required')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);gpu=gpu_gate(cfg,a.gpu_uuid)
    if sha(a.frontend)!=FRONTEND_SHA256:raise ValueError('Wrong original frontend')
    original=torch.load(a.frontend,map_location='cpu',weights_only=False);model=VisionEncoder().cuda();model.load_state_dict(original['model'],strict=True);torch.nn.init.xavier_uniform_(model.head.weight);torch.nn.init.zeros_(model.head.bias)
    if sum(v.numel() for v in model.features.parameters())!=32128 or sum(v.numel() for v in model.head.parameters())!=3096:raise ValueError('Changed original architecture')
    train=load_arrays(a.data,'train');val=load_arrays(a.data,'validation');tx=torch.from_numpy(train['images']).cuda();vx=torch.from_numpy(val['images']).cuda();y=torch.as_tensor(train['labels'],device='cuda');a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();best=(-1.,-1.,-float('inf'));last_improvement=0;history=[];best_epoch=0
    optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=cfg['feature_lr']),dict(params=model.head.parameters(),lr=cfg['temporary_head_lr'])],weight_decay=cfg['weight_decay'])
    put(a.out/'protocol.json',dict(config=cfg,original_frontend_SHA256=FRONTEND_SHA256,TRAIN=6000,validation=2000,TRAIN_locations=3000,shared_pretraining_domains=['RGB','SAR'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),data_manifest_SHA256=sha(a.data/'manifest.json'),TEST_access=False,SDK=False,gpu=gpu,optical_result=False,automatically_replace_optical_frontend=False))
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;correct=0;seen=0;factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs']-1)))
        for group,lr in zip(optimizer.param_groups,(cfg['feature_lr'],cfg['temporary_head_lr'])):group['lr']=lr*factor
        for ix in balanced_batches(train,[0,1],cfg['batch_size'],cfg['seed']+epoch):
            x=tx[ix].clone();h=torch.rand(len(x),device='cuda')<cfg['TRAIN_horizontal_flip_probability'];v=torch.rand(len(x),device='cuda')<cfg['TRAIN_vertical_flip_probability'];x[h]=x[h].flip(2);x[v]=x[v].flip(1)
            logits=model(x)[1][:,:10];cost=F.cross_entropy(logits,y[ix],label_smoothing=cfg['label_smoothing']);optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(cost):raise RuntimeError('Nonfinite auxiliary TRAIN objective')
            cost.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(cost.detach())*len(ix);seen+=len(ix);correct+=int((logits.argmax(1)==y[ix]).sum())
        scores=score(model,vx,val);value=selection(scores,[0,1]);row=dict(epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy=correct/seen,validation=scores['scores']);history.append(row)
        checkpoint=dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},config=dict(original['config'],feature_parameters=32128,temporary_head_parameters=3096,test_policy='never read',T20_domain_pretraining=cfg),epoch=epoch,metrics=row,diagnostic_only=True,TEST_access=False,original_frontend_SHA256=FRONTEND_SHA256)
        torch.save(checkpoint,a.out/'last_checkpoint.pt')
        if value>best:best=value;best_epoch=epoch;last_improvement=epoch;torch.save(checkpoint,a.out/'best_checkpoint.pt')
        put(a.out/'history.json',history);put(a.out/'status.json',dict(state='training',**row,best_epoch=best_epoch,updated=time.time()))
        if time.monotonic()-started>cfg['max_seconds'] or (epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']):break
    assert sha(a.frontend)==FRONTEND_SHA256
    put(a.out/'completion.json',dict(passed=True,optical_result=False,best_epoch=best_epoch,epochs_completed=epoch,best_validation_mean=best[0],best_SHA256=sha(a.out/'best_checkpoint.pt'),last_SHA256=sha(a.out/'last_checkpoint.pt'),TEST_access=False,SDK=False,automatically_replace_optical_frontend=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete',best_epoch=best_epoch,updated=time.time()))


if __name__=='__main__':main()
