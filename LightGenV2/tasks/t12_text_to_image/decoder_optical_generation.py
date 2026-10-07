"""Text-directed image generation with optics in the first decoder block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .chair_style_transfer import ChairStyleDiscriminator, ResidualDown
from .modeling import CompactFourierOptics, ConditionedElectronicResidual, ScaleMatchedFusion


@dataclass(frozen=True)
class DecoderOpticalConfig:
    task: str
    image_size: int
    text_dim: int
    condition_dim: int
    widths: tuple[int, ...]
    optical: bool
    experts: int
    top_k: int
    alpha_initial: float
    alpha_minimum: float
    alpha_maximum: float
    rms_epsilon: float
    residual_limit: float
    maximum_skip: float
    view_flow_limit: float
    view_warp_mix: float
    view_residual_limit: float
    batch_size: int
    epochs: int
    learning_rate: float
    phase_learning_rate: float
    weight_decay: float
    num_workers: int
    amp: bool
    adversarial_weight: float
    reconstruction_weight: float
    edge_weight: float
    background_weight: float
    adversarial_warmup_epochs: int
    sample_every_epochs: int

    def validate(self) -> None:
        if self.task not in {"style", "view"}: raise ValueError("task must be style or view")
        if self.image_size != 128 or len(self.widths) != 4: raise ValueError("Expected 128px and four widths")
        if self.optical and self.alpha_minimum < 0.4: raise ValueError("Decoder optics require alpha_minimum >= 0.4")
        if not 0 <= self.alpha_minimum < self.alpha_initial < self.alpha_maximum <= 1: raise ValueError("Invalid alpha range")
        if not 0 < self.view_flow_limit <= .26 or not 0 < self.view_warp_mix <= 1 or not 0 <= self.view_residual_limit <= .1: raise ValueError("Invalid structure-preserving view limits")
        if self.top_k > self.experts: raise ValueError("top_k exceeds experts")


def load_decoder_optical_config(path: str | Path) -> DecoderOpticalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")); m=raw["model"]; f=raw["fusion"]; t=raw["training"]; loss=raw["loss"]
    config = DecoderOpticalConfig(
        task=str(m["task"]), image_size=int(m["image_size"]), text_dim=int(m["text_dim"]),
        condition_dim=int(m["condition_dim"]), widths=tuple(int(x) for x in m["widths"]), optical=bool(m["optical"]),
        experts=int(m["experts"]), top_k=int(m["top_k"]), alpha_initial=float(f["alpha_initial"]),
        alpha_minimum=float(f["alpha_minimum"]), alpha_maximum=float(f["alpha_maximum"]), rms_epsilon=float(f["rms_epsilon"]),
        residual_limit=float(m["residual_limit"]), maximum_skip=float(m["maximum_skip"]),
        view_flow_limit=float(m.get("view_flow_limit",.26)), view_warp_mix=float(m.get("view_warp_mix",1.0)),
        view_residual_limit=float(m.get("view_residual_limit",.06)),
        batch_size=int(t["batch_size"]), epochs=int(t["epochs"]), learning_rate=float(t["learning_rate"]),
        phase_learning_rate=float(t["phase_learning_rate"]), weight_decay=float(t["weight_decay"]),
        num_workers=int(t["num_workers"]), amp=bool(t["amp"]), adversarial_weight=float(loss["adversarial_weight"]),
        reconstruction_weight=float(loss["reconstruction_weight"]), edge_weight=float(loss["edge_weight"]),
        background_weight=float(loss["background_weight"]), adversarial_warmup_epochs=int(t["adversarial_warmup_epochs"]),
        sample_every_epochs=int(t["sample_every_epochs"]),
    ); config.validate(); return config


class DecoderHybridGeneratorBlock(nn.Module):
    """First decoder block: same decoder state enters E and O in parallel."""
    def __init__(self, width: int, condition_dim: int, grid: int, config: DecoderOpticalConfig) -> None:
        super().__init__(); self.width=width; self.grid=grid; self.optical_enabled=config.optical
        self.electronic = ConditionedElectronicResidual(width, condition_dim, grid)
        if config.optical:
            self.optical_condition = nn.Linear(condition_dim, 2*width)
            self.optical = CompactFourierOptics(width, grid, config.experts, config.top_k)
            self.expert_gate = nn.Parameter(torch.tensor(-1.0)); self.global_gate = nn.Parameter(torch.tensor(-1.0))
            self.fusion = ScaleMatchedFusion(config.alpha_initial, config.alpha_minimum, config.alpha_maximum, config.rms_epsilon)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        tokens=value.flatten(2).transpose(1,2); electronic=self.electronic(tokens,condition)
        if not self.optical_enabled: return electronic.transpose(1,2).reshape_as(value)
        scale,shift=self.optical_condition(condition).chunk(2,dim=1); optical_input=tokens*(1+scale[:,None])+shift[:,None]
        stage1=tokens+torch.sigmoid(self.expert_gate)*self.optical.expert(optical_input)
        optical=stage1+torch.sigmoid(self.global_gate)*self.optical.global_block(stage1)
        return self.fusion(electronic,optical).transpose(1,2).reshape_as(value)


class GatedDecoderUp(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, condition_dim: int, maximum_skip: float) -> None:
        super().__init__(); self.maximum_skip=float(maximum_skip)
        self.skip_gate=nn.Linear(condition_dim,skip_channels); self.conv1=nn.Conv2d(input_channels+skip_channels,output_channels,3,padding=1)
        self.conv2=nn.Conv2d(output_channels,output_channels,3,padding=1); groups=min(16,output_channels)
        while output_channels%groups: groups-=1
        self.norm1=nn.GroupNorm(groups,output_channels,affine=False); self.norm2=nn.GroupNorm(groups,output_channels,affine=False)
        self.affine=nn.Linear(condition_dim,4*output_channels)

    def forward(self,value:torch.Tensor,skip:torch.Tensor,condition:torch.Tensor)->torch.Tensor:
        value=F.interpolate(value,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        gate=self.maximum_skip*torch.sigmoid(self.skip_gate(condition))[:,:,None,None]
        value=torch.cat((value,skip*gate),dim=1); s1,b1,s2,b2=self.affine(condition).chunk(4,dim=1)
        value=F.silu(self.norm1(self.conv1(value))*(1+s1[:,:,None,None])+b1[:,:,None,None])
        return F.silu(self.norm2(self.conv2(value))*(1+s2[:,:,None,None])+b2[:,:,None,None])


class DecoderOpticalGenerator(nn.Module):
    def __init__(self, config: DecoderOpticalConfig) -> None:
        super().__init__(); self.config=config; w=config.widths
        self.condition=nn.Sequential(nn.LayerNorm(config.text_dim),nn.Linear(config.text_dim,config.condition_dim),nn.SiLU(),nn.Linear(config.condition_dim,config.condition_dim),nn.SiLU())
        self.stem=nn.Sequential(nn.Conv2d(3,w[0],3,padding=1),nn.SiLU()); self.down1=ResidualDown(w[0],w[1]); self.down2=ResidualDown(w[1],w[2]); self.down3=ResidualDown(w[2],w[3])
        self.decoder_generator=DecoderHybridGeneratorBlock(w[3],config.condition_dim,config.image_size//8,config)
        self.up3=GatedDecoderUp(w[3],w[2],w[2],config.condition_dim,config.maximum_skip)
        self.up2=GatedDecoderUp(w[2],w[1],w[1],config.condition_dim,config.maximum_skip)
        self.up1=GatedDecoderUp(w[1],w[0],w[0],config.condition_dim,config.maximum_skip)
        output_channels=3 if config.task=="style" else 5
        self.to_rgb=nn.Sequential(nn.Conv2d(w[0],w[0],3,padding=1),nn.SiLU(),nn.Conv2d(w[0],output_channels,3,padding=1))
        nn.init.zeros_(self.to_rgb[-1].weight); nn.init.zeros_(self.to_rgb[-1].bias)

    @staticmethod
    def foreground_mask(reference:torch.Tensor)->torch.Tensor:
        rgb=reference.float().add(1).mul(.5); return (((1-rgb).amax(1,keepdim=True)-.06)/.16).clamp(0,1)

    def forward_with_aux(self,reference:torch.Tensor,text:torch.Tensor)->tuple[torch.Tensor,dict[str,torch.Tensor]]:
        condition=self.condition(text.float()); s0=self.stem(reference); s1=self.down1(s0); s2=self.down2(s1); encoded=self.down3(s2)
        value=self.decoder_generator(encoded,condition); value=self.up3(value,s2,condition); value=self.up2(value,s1,condition); value=self.up1(value,s0,condition)
        raw=self.to_rgb(value)
        if self.config.task=="style":
            delta=torch.tanh(raw)*self.config.residual_limit; mask=self.foreground_mask(reference).to(delta.dtype); output=(reference+mask*delta).clamp(-1,1)
        else:
            batch,_,height,width=reference.shape
            yy,xx=torch.meshgrid(torch.linspace(-1,1,height,device=reference.device,dtype=raw.dtype),torch.linspace(-1,1,width,device=reference.device,dtype=raw.dtype),indexing="ij")
            base=torch.stack((xx,yy),dim=-1)[None].expand(batch,-1,-1,-1)
            flow=self.config.view_flow_limit*torch.tanh(raw[:,:2]).permute(0,2,3,1)
            warped=F.grid_sample(reference,base+flow,mode="bilinear",padding_mode="border",align_corners=True)
            mask=F.max_pool2d(self.foreground_mask(warped).to(raw.dtype),11,stride=1,padding=5)
            candidate=reference+self.config.view_warp_mix*(warped-reference)
            delta=self.config.view_residual_limit*torch.tanh(raw[:,2:]); output=(candidate+mask*delta).clamp(-1,1)
        return output,{"encoded":encoded,"decoder_generated":value,"delta":delta,"mask":mask,"flow":flow if self.config.task=="view" else torch.zeros((),device=reference.device)}

    def forward(self,reference:torch.Tensor,text:torch.Tensor)->torch.Tensor: return self.forward_with_aux(reference,text)[0]


def architecture_report(model:DecoderOpticalGenerator,discriminator:nn.Module|None=None)->dict[str,Any]:
    generator=sum(p.numel() for p in model.parameters()); disc=0 if discriminator is None else sum(p.numel() for p in discriminator.parameters())
    optical=sum(p.numel() for n,p in model.named_parameters() if "decoder_generator.optical" in n)
    alpha=None if not model.config.optical else float(model.decoder_generator.fusion.alpha.detach())
    return {"variant":f"decoder_optical_{model.config.task}_{'hybrid' if model.config.optical else 'electronic'}","generator_parameters":generator,"discriminator_parameters":disc,"total_training_parameters":generator+disc,"decoder_optical_parameters":optical,"optics_location":"first_decoder_generator_block" if model.config.optical else None,"alpha":alpha,"alpha_minimum":model.config.alpha_minimum if model.config.optical else None,"view_flow_limit":model.config.view_flow_limit if model.config.task=="view" else None,"view_warp_mix":model.config.view_warp_mix if model.config.task=="view" else None,"single_pass":True,"decoder_calls":1,"gan_training":True}


def config_payload(config:DecoderOpticalConfig)->dict[str,Any]:
    value=asdict(config); value["widths"]=list(config.widths); return value


__all__=["ChairStyleDiscriminator","DecoderOpticalConfig","DecoderOpticalGenerator","architecture_report","config_payload","load_decoder_optical_config"]
