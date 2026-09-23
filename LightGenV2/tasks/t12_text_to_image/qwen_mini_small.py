"""Sub-50M editor that keeps a real, width-distilled Qwen language front-end.

The shared Qwen tokenizer and token embedding table are frozen and excluded
from the counted task parameters by the project convention.  Two trainable
Qwen-style causal Transformer blocks remain inside every task checkpoint.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .feature_cache import _qwen_prompts
from .half_qwen import load_half_qwen_text_encoder
from .small_fullframe import (
    PromptPairDataset, SmallEditorConfig, SmallFullFrameEditor, _edge,
    build_dataset,
)


@dataclass(frozen=True)
class QwenMiniConfig:
    input_width: int = 2048
    width: int = 768
    intermediate_width: int = 2048
    layers: int = 2
    heads: int = 12
    condition_dim: int = 160
    max_length: int = 64


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__(); self.weight = nn.Parameter(torch.ones(width)); self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps).to(value.dtype) * self.weight


def _rope(value: torch.Tensor) -> torch.Tensor:
    # value: B,H,L,D.  Qwen uses rotary position encoding inside attention.
    length, width = value.shape[-2:]
    if width % 2:
        raise ValueError("Rotary head width must be even")
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width))
    angles = positions[:, None] * inverse[None]
    cosine, sine = angles.cos().to(value.dtype), angles.sin().to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), dim=-1).flatten(-2)


class QwenMiniBlock(nn.Module):
    """Qwen-compatible pre-norm causal attention plus gated SwiGLU MLP."""

    def __init__(self, config: QwenMiniConfig) -> None:
        super().__init__(); width=config.width; self.heads=config.heads; self.head_width=width//config.heads
        if width % config.heads or self.head_width % 2:
            raise ValueError("Qwen-mini width must split into even rotary heads")
        self.input_layernorm=RMSNorm(width)
        self.q_proj=nn.Linear(width,width,bias=False); self.k_proj=nn.Linear(width,width,bias=False)
        self.v_proj=nn.Linear(width,width,bias=False); self.o_proj=nn.Linear(width,width,bias=False)
        self.post_attention_layernorm=RMSNorm(width)
        self.gate_proj=nn.Linear(width,config.intermediate_width,bias=False)
        self.up_proj=nn.Linear(width,config.intermediate_width,bias=False)
        self.down_proj=nn.Linear(config.intermediate_width,width,bias=False)

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residual=value; norm=self.input_layernorm(value); batch,length,_=norm.shape
        reshape=lambda x: x.view(batch,length,self.heads,self.head_width).transpose(1,2)
        query=_rope(reshape(self.q_proj(norm))); key=_rope(reshape(self.k_proj(norm))); values=reshape(self.v_proj(norm))
        causal=torch.ones(length,length,dtype=torch.bool,device=value.device).tril()
        allowed=causal[None,None] & valid[:,None,None,:]
        attention=F.scaled_dot_product_attention(query,key,values,attn_mask=allowed,dropout_p=0.0)
        value=residual+self.o_proj(attention.transpose(1,2).reshape(batch,length,-1))
        norm=self.post_attention_layernorm(value)
        return value+self.down_proj(F.silu(self.gate_proj(norm))*self.up_proj(norm))


class QwenMiniTextEncoder(nn.Module):
    def __init__(self, config: QwenMiniConfig) -> None:
        super().__init__(); self.config=config
        self.input_projection=nn.Linear(config.input_width,config.width,bias=False)
        self.layers=nn.ModuleList(QwenMiniBlock(config) for _ in range(config.layers))
        self.norm=RMSNorm(config.width); self.condition_projection=nn.Linear(config.width,config.condition_dim)

    def hidden(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value=self.input_projection(embeddings); valid=mask.bool()
        for layer in self.layers: value=layer(value,valid)
        value=self.norm(value); last=valid.long().sum(1).clamp_min(1)-1
        return value[torch.arange(len(value),device=value.device),last]

    def forward(self, inputs: tuple[torch.Tensor,torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs,tuple):
            raise TypeError("QwenMiniTextEncoder expects (frozen_token_embeddings, attention_mask)")
        return self.condition_projection(self.hidden(*inputs))


@torch.inference_mode()
def build_qwen_embedding_cache(
    *, instruction_cache: Path, qwen_checkpoint: Path, output: Path,
    device: torch.device, max_length: int = 64, force: bool = False,
) -> dict[str, Any]:
    """Cache frozen Qwen token embeddings; the table itself stays shared."""
    if output.exists() and not force:
        return torch.load(output,map_location="cpu",weights_only=False)["meta"]
    payload=torch.load(instruction_cache,map_location="cpu",weights_only=False)
    prompts=[row["prompt"] for row in payload["rows"]]
    language,processor,qwen_report=load_half_qwen_text_encoder(qwen_checkpoint,device,keep_layers=1)
    encoded=_qwen_prompts(processor,prompts)
    ids=encoded["input_ids"][:,-max_length:].to(device); mask=encoded["attention_mask"][:,-max_length:]
    embeddings=language.embed_tokens(ids).half().cpu()
    result={
        "meta":{"schema_version":1,"prompts":len(prompts),"max_length":max_length,
                "input_width":embeddings.shape[-1],"shared_qwen_embedding":qwen_report["token_embedding_parameters_excluded_by_project_convention"],
                "qwen_checkpoint":str(qwen_checkpoint)},
        "prompts":prompts,"embeddings":embeddings,"attention_mask":mask.bool().cpu(),
        "teacher_pooled":payload["text"].float().cpu(),
    }
    output.parent.mkdir(parents=True,exist_ok=True); temporary=output.with_suffix(".tmp")
    torch.save(result,temporary); temporary.replace(output)
    del language,processor
    if device.type=="cuda": torch.cuda.empty_cache()
    return result["meta"]


class PromptEmbeddingLookup:
    def __init__(self,path:Path) -> None:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        self.lookup={prompt:index for index,prompt in enumerate(payload["prompts"])}
        self.embeddings=payload["embeddings"]; self.mask=payload["attention_mask"]
        self.teacher=payload["teacher_pooled"]

    def batch(self,prompts:list[str] | tuple[str,...],device:torch.device):
        indices=torch.as_tensor([self.lookup[prompt] for prompt in prompts],dtype=torch.long)
        return self.embeddings[indices].to(device),self.mask[indices].to(device),self.teacher[indices].to(device)


def _condition_indices(batch:dict[str,Any],device:torch.device)->torch.Tensor:
    return batch["condition_index"].to(device)


@torch.inference_mode()
def evaluate(model,loader,lookup,device,seed):
    model.eval();totals={"mse":0.,"l1":0.,"edge_l1":0.,"copy_mse":0.};count=0;generator=torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        reference=batch["reference"].to(device);target=batch["target"].to(device);embeddings,mask,_=lookup.batch(list(batch["prompt"]),device)
        noise=torch.randn(reference.shape,generator=generator,device=device);controls=_condition_indices(batch,device) if model.control_embedding is not None else None
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): output=model(reference,(embeddings,mask),noise,controls)
        size=len(reference);totals["mse"]+=float(F.mse_loss(output,target))*size;totals["l1"]+=float(F.l1_loss(output,target))*size
        totals["edge_l1"]+=float(F.l1_loss(_edge(output),_edge(target)))*size;totals["copy_mse"]+=float(F.mse_loss(reference,target))*size;count+=size
    result={key:value/count for key,value in totals.items()};result["mse_improvement_over_copy"]=1-result["mse"]/max(result["copy_mse"],1e-8);return result


@torch.inference_mode()
def save_samples(model,dataset:PromptPairDataset,lookup,output:Path,device,seed)->None:
    model.eval();count=min(12,len(dataset));step=max(1,len(dataset)//count);items=[dataset[min(i*step,len(dataset)-1)] for i in range(count)]
    reference=torch.stack([x["reference"] for x in items]).to(device);embeddings,mask,_=lookup.batch([x["prompt"] for x in items],device)
    generator=torch.Generator(device=device).manual_seed(seed);noise=torch.randn(reference.shape,generator=generator,device=device)
    controls=torch.as_tensor([x["condition_index"] for x in items],device=device) if model.control_embedding is not None else None
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):prediction=model(reference,(embeddings,mask),noise,controls)
    cell=model.config.image_size;label=300;canvas=Image.new("RGB",(label+3*cell,count*cell),"white");draw=ImageDraw.Draw(canvas)
    for row,item in enumerate(items):
        for column,value in enumerate((item["reference"],item["target"],prediction[row].cpu())):
            array=value.add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy();canvas.paste(Image.fromarray(array),(label+column*cell,row*cell))
        draw.text((4,row*cell+4),item["prompt"][:47],fill="black");draw.text((4,row*cell+28),"input | target | Qwen-mini student",fill="black")
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output,quality=94,subsampling=0)


def train_qwen_mini_editor(
    *,task:str,data_dir:Path,instruction_cache:Path,embedding_cache:Path,output_dir:Path,
    device:torch.device,epochs:int=20,batch_size:int=16,learning_rate:float=2e-4,seed:int=42,
)->dict[str,Any]:
    if output_dir.exists():raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    editor_config=SmallEditorConfig(control_classes=64);text_config=QwenMiniConfig(condition_dim=editor_config.condition_dim)
    datasets={name:build_dataset(task,data_dir,name,editor_config.image_size,instruction_cache) for name in ("train","val","test")}
    loaders={name:DataLoader(value,batch_size=batch_size,shuffle=name=="train",num_workers=4,pin_memory=True,persistent_workers=True) for name,value in datasets.items()}
    lookup=PromptEmbeddingLookup(embedding_cache);model=SmallFullFrameEditor(editor_config);model.text=QwenMiniTextEncoder(text_config);model=model.to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False);parameters=sum(p.numel() for p in model.parameters())
    if parameters>=50_000_000:raise AssertionError(parameters)
    teacher_head=nn.Linear(text_config.width,lookup.teacher.shape[1]).to(device);classifier=nn.Linear(editor_config.condition_dim,64).to(device)
    optimizer=torch.optim.AdamW([*model.parameters(),*teacher_head.parameters(),*classifier.parameters()],lr=learning_rate,weight_decay=1e-2)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");initial=evaluate(model,loaders["val"],lookup,device,seed+100)
    best=math.inf;best_epoch=0;history=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();teacher_head.train();classifier.train();total=samples=0
        for batch in loaders["train"]:
            reference=batch["reference"].to(device);target=batch["target"].to(device);embeddings,mask,teacher=lookup.batch(list(batch["prompt"]),device);noise=torch.randn_like(reference);labels=_condition_indices(batch,device)
            if random.random()<.5:reference=reference.flip(-1);target=target.flip(-1);noise=noise.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                output=model(reference,(embeddings,mask),noise,labels);hidden=model.text.hidden(embeddings,mask);condition=model.text.condition_projection(hidden)
                image_loss=F.l1_loss(output,target)+.65*F.mse_loss(output,target)+.10*F.l1_loss(_edge(output),_edge(target))
                distill=1-F.cosine_similarity(teacher_head(hidden).float(),teacher.float(),dim=-1).mean()
                loss=image_loss+.15*distill+.20*F.cross_entropy(classifier(condition).float(),labels)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([*model.parameters(),*teacher_head.parameters(),*classifier.parameters()],3.0);scaler.step(optimizer);scaler.update()
            with torch.no_grad():
                for destination,source in zip(ema.parameters(),model.parameters()):destination.lerp_(source,.01)
                for destination,source in zip(ema.buffers(),model.buffers()):destination.copy_(source)
            total+=float(loss.detach())*len(reference);samples+=len(reference)
        validation=evaluate(ema,loaders["val"],lookup,device,seed+epoch);row={"epoch":epoch,"train_loss":total/samples,"validation":validation,"alpha":float(ema.bottleneck.fusion.alpha)};history.append(row);print(json.dumps(row),flush=True)
        if validation["mse"]<best:
            best=validation["mse"];best_epoch=epoch;torch.save({"schema_version":2,"task":task,"editor_config":{**asdict(editor_config),"widths":list(editor_config.widths)},"qwen_mini_config":asdict(text_config),"model":{key:value.detach().half().cpu() for key,value in ema.state_dict().items()},"counted_parameters":parameters,"shared_qwen_token_embedding_excluded":True,"qwen_transformer_layers":2,"epoch":epoch,"validation":validation,"hard_pixel_composite":False,"gan_used":False,"inference_iterations":1},output_dir/"best_model.pt")
        if epoch in {1,5,10,epochs}:save_samples(ema,datasets["val"],lookup,output_dir/"samples"/f"epoch_{epoch:03d}.jpg",device,seed+epoch)
    payload=torch.load(output_dir/"best_model.pt",map_location="cpu",weights_only=False);ema.load_state_dict(payload["model"]);test=evaluate(ema,loaders["test"],lookup,device,seed+999);save_samples(ema,datasets["test"],lookup,output_dir/"final_grid.jpg",device,seed+999)
    report={"schema_version":2,"task":task,"best_epoch":best_epoch,"parameters":parameters,"under_50m":parameters<50_000_000,"text_frontend":"shared frozen Qwen tokenizer/embedding -> 2048-to-768 projection -> 2 Qwen-style blocks","qwen_transformer_layers":2,"initial_validation":initial,"test":test,"history":history,"training_seconds":time.perf_counter()-started,"optical_alpha":float(ema.bottleneck.fusion.alpha),"hard_pixel_composite":False,"gan_used":False,"inference_iterations":1,"resolution":editor_config.image_size}
    (output_dir/"training_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    del model,ema,optimizer,teacher_head,classifier
    if device.type=="cuda":torch.cuda.empty_cache()
    return report


__all__=["QwenMiniConfig","QwenMiniTextEncoder","build_qwen_embedding_cache","train_qwen_mini_editor"]
