"""Training for backpack styling and turntable view synthesis."""

from __future__ import annotations

import copy,json,random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader,default_collate

from .dataset import validate_split_contract
from .decoder_optical_generation import (
    ChairStyleDiscriminator,DecoderOpticalConfig,DecoderOpticalGenerator,architecture_report,config_payload,
)
from .product_instruction_data import BackpackStyleDataset,TurntableViewDataset


def _seed(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _dataset(task:str,data:Path,split:str,size:int,cache:Path):
    return BackpackStyleDataset(data,split,size,cache) if task=="style" else TurntableViewDataset(data,split,size,cache)


def _loader(task:str,data:Path,split:str,config:DecoderOpticalConfig,cache:Path,seed:int,shuffle:bool)->DataLoader:
    return DataLoader(_dataset(task,data,split,config.image_size,cache),batch_size=config.batch_size,shuffle=shuffle,num_workers=config.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=config.num_workers>0,drop_last=shuffle,generator=torch.Generator().manual_seed(seed))


def _edge(value:torch.Tensor)->torch.Tensor:
    gray=value.float().mean(1,keepdim=True);kx=value.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).reshape(1,1,3,3);ky=kx.transpose(-1,-2)
    return torch.sqrt(F.conv2d(gray,kx,padding=1).square()+F.conv2d(gray,ky,padding=1).square()+1e-6)


def _foreground(value:torch.Tensor)->torch.Tensor:
    return DecoderOpticalGenerator.foreground_mask(value)


def _losses(output:torch.Tensor,target:torch.Tensor,reference:torch.Tensor,config:DecoderOpticalConfig)->dict[str,torch.Tensor]:
    mask=_foreground(target); weight=1+5*mask
    full=((output.float()-target.float()).abs()*weight).sum()/(weight.sum()*3)
    multi=[F.l1_loss(F.adaptive_avg_pool2d(output.float(),(s,s)),F.adaptive_avg_pool2d(target.float(),(s,s))) for s in (32,64,128)]
    reconstruction=.5*full+.5*torch.stack(multi).mean(); edge=(_edge(output)-_edge(target)).abs().mean()
    background=((output.float()-target.float()).abs()*(1-mask)).mean()
    return {"reconstruction":reconstruction,"edge":edge,"background":background}


@torch.no_grad()
def _update(ema:torch.nn.Module,model:torch.nn.Module,beta:float=.995)->None:
    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-beta)
    for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)


@torch.no_grad()
def evaluate(model:DecoderOpticalGenerator,loader:DataLoader,device:torch.device)->dict[str,float]:
    model.eval();tot={"l1":0.,"edge":0.,"background":0.};count=0
    for batch in loader:
        ref=batch["reference"].to(device);target=batch["target"].to(device);text=batch["text"].to(device);out=model(ref,text);n=len(ref);count+=n
        mask=_foreground(target);tot["l1"]+=float(F.l1_loss(out.float(),target.float()))*n;tot["edge"]+=float((_edge(out)-_edge(target)).abs().mean())*n;tot["background"]+=float(((out-target).abs()*(1-mask)).mean())*n
    return {k:v/count for k,v in tot.items()}


def _pil(value:torch.Tensor)->Image.Image:
    array=value.detach().float().add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy();return Image.fromarray(array)


@torch.no_grad()
def save_grid(model:DecoderOpticalGenerator,dataset,output:Path,device:torch.device)->None:
    variants=3;conditions=6 if model.config.task=="style" else 4;indices=[]
    for source in range(2):
        base=source*len(dataset.prompts)
        indices.extend(base+c*variants for c in range(conditions))
    batch=default_collate([dataset[i] for i in indices]);ref=batch["reference"].to(device);target=batch["target"].to(device);pred=model(ref,batch["text"].to(device))
    tile=128;head=20;rows=len(indices);canvas=Image.new("RGB",(3*tile,rows*(tile+head)),"white");draw=ImageDraw.Draw(canvas)
    for r in range(rows):
        for c,(name,value) in enumerate((("input",ref[r]),("target",target[r]),("prediction",pred[r]))):
            x=c*tile;y=r*(tile+head);draw.text((x+3,y+3),name,fill="black");canvas.paste(_pil(value),(x,y+head))
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)


def _warmstart(model:DecoderOpticalGenerator,disc:ChairStyleDiscriminator,path:Path)->dict[str,Any]:
    payload=torch.load(path,map_location="cpu",weights_only=False);report=model.load_state_dict(payload["generator_ema"],strict=False)
    allowed=("decoder_generator.optical","decoder_generator.fusion","decoder_generator.expert_gate","decoder_generator.global_gate")
    bad=[x for x in report.missing_keys if not x.startswith(allowed)]
    if bad or report.unexpected_keys:raise RuntimeError(f"unsafe warmstart missing={bad} unexpected={report.unexpected_keys}")
    disc.load_state_dict(payload["discriminator"]);return {"source":str(path),"epoch":payload["epoch"],"missing":list(report.missing_keys)}


def train_decoder_optical(data_dir:Path,cache:Path,run_dir:Path,config:DecoderOpticalConfig,device:torch.device,*,seed:int=113,initialize:Path|None=None)->dict[str,Any]:
    _seed(seed);validate_split_contract(data_dir);run_dir.mkdir(parents=True,exist_ok=False)
    train_loader=_loader(config.task,data_dir,"train",config,cache,seed,True);val_loader=_loader(config.task,data_dir,"val",config,cache,seed+1,False)
    model=DecoderOpticalGenerator(config).to(device);disc=ChairStyleDiscriminator(config.text_dim).to(device);warm=None
    if initialize is not None:warm=_warmstart(model,disc,initialize)
    ema=copy.deepcopy(model).eval().requires_grad_(False);regular=[];optical=[]
    for name,p in model.named_parameters():(optical if "decoder_generator.optical" in name else regular).append(p)
    groups=[{"params":regular,"lr":config.learning_rate}]
    if optical:groups.append({"params":optical,"lr":config.phase_learning_rate})
    go=torch.optim.AdamW(groups,weight_decay=config.weight_decay,betas=(.5,.99));do=torch.optim.AdamW(disc.parameters(),lr=config.learning_rate,betas=(.5,.99))
    arch=architecture_report(model,disc);(run_dir/"architecture.json").write_text(json.dumps(arch,indent=2)+"\n",encoding="utf-8")
    history=[];best=float("inf");best_epoch=0;autocast=dict(device_type=device.type,dtype=torch.bfloat16,enabled=config.amp and device.type=="cuda")
    for epoch in range(1,config.epochs+1):
        model.train();disc.train();tot={};seen=0;adv=config.adversarial_weight*min(1.,epoch/max(1,config.adversarial_warmup_epochs))
        for batch in train_loader:
            ref=batch["reference"].to(device,non_blocking=True);target=batch["target"].to(device,non_blocking=True);text=batch["text"].to(device,non_blocking=True)
            if torch.rand(())<.5:ref,target=ref.flip(-1),target.flip(-1)
            with torch.autocast(**autocast):out=model(ref,text)
            for p in disc.parameters():p.requires_grad_(True)
            do.zero_grad(set_to_none=True)
            with torch.autocast(**autocast):real=disc(target,text);fake=disc(out.detach(),text);dl=F.relu(1-real.float()).mean()+F.relu(1+fake.float()).mean()
            dl.backward();do.step()
            for p in disc.parameters():p.requires_grad_(False)
            go.zero_grad(set_to_none=True)
            with torch.autocast(**autocast):parts=_losses(out,target,ref,config);score=disc(out,text);gl=config.reconstruction_weight*parts["reconstruction"]+config.edge_weight*parts["edge"]+config.background_weight*parts["background"]-adv*score.float().mean()
            gl.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);go.step();_update(ema,model)
            n=len(ref);seen+=n;values={"generator":float(gl.detach()),"discriminator":float(dl.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
            for k,v in values.items():tot[k]=tot.get(k,0.)+n*v
        val=evaluate(ema,val_loader,device);selection=val["l1"]+.15*val["edge"];record={"epoch":epoch,"train":{k:v/seen for k,v in tot.items()},"validation":val,"selection":selection};history.append(record);print(json.dumps(record),flush=True)
        payload={"epoch":epoch,"generator":model.state_dict(),"generator_ema":ema.state_dict(),"discriminator":disc.state_dict(),"config":config_payload(config),"architecture":arch,"validation":val,"warmstart":warm}
        torch.save(payload,run_dir/"latest_checkpoint.pt")
        if selection<best:best=selection;best_epoch=epoch;torch.save(payload,run_dir/"best_checkpoint.pt")
        if epoch==1 or epoch%config.sample_every_epochs==0 or epoch==config.epochs:save_grid(ema,val_loader.dataset,run_dir/f"sample_epoch_{epoch:04d}.png",device)
        (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
    summary={"best_epoch":best_epoch,"best_selection":best,"architecture":arch,"warmstart":warm};(run_dir/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");return summary


__all__=["evaluate","save_grid","train_decoder_optical"]
