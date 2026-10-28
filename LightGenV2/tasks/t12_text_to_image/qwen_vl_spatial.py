"""Qwen-only image conditioning: semantic tokens plus a spatial visual readout.

Reference VAE latents are TRAIN supervision only. No RGB bypass, target, mask,
catalogue index or reference VAE encoder is an input of the deployed forward.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .qwen_vl_turbo_model import NativeTurboHead,QueryResampler
from .qwen_vl_turbo_training import configure_unet_training


class SpatialBridge(nn.Module):
    def __init__(self,input_dim=2048,width=128,channels=16,grid=8,conditional=False):
        super().__init__()
        self.grid=grid
        self.norm=nn.LayerNorm(input_dim)
        self.project=nn.Linear(input_dim,width)
        self.up1=nn.Conv2d(width,width//2,3,padding=1)
        self.up2=nn.Conv2d(width//2,width//4,3,padding=1)
        self.output=nn.Conv2d(width//4,channels,1)
        self.aux_reference=nn.Conv2d(width//4,4,1)
        self.film=nn.Linear(input_dim,2*width) if conditional else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)

    def decode_features(self,x):
        x=F.silu(self.up1(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
        return F.silu(self.up2(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))

    def forward(self,visual_tokens,with_aux=False,context=None):
        if visual_tokens.ndim!=3 or visual_tokens.shape[1]!=self.grid**2:
            raise ValueError('Expected one verified, row-major Qwen visual grid')
        x=self.project(self.norm(visual_tokens.float())).transpose(1,2)
        x=x.reshape(len(x),-1,self.grid,self.grid)
        auxiliary=None
        if self.film is not None:
            if context is None or context.shape!=(len(x),self.film.in_features):raise ValueError('Instruction context required by conditional spatial readout')
            if with_aux:
                if self.aux_reference is None:raise ValueError('TRAIN-only auxiliary readout was discarded')
                auxiliary=self.aux_reference(self.decode_features(x))
            gamma,beta=self.film(F.layer_norm(context.float(),(context.shape[-1],))).chunk(2,-1)
            rms=x.float().square().mean((1,2,3),keepdim=True).sqrt().detach().clamp_min(1e-6)
            x=x*(1+gamma.tanh()[:,:,None,None])+.5*beta.tanh()[:,:,None,None]*rms
        x=self.decode_features(x)
        spatial=self.output(x)
        if with_aux:
            if self.aux_reference is None:raise ValueError('TRAIN-only auxiliary readout was discarded')
            return spatial,auxiliary if auxiliary is not None else self.aux_reference(x)
        return spatial


class SpatialInputConv(nn.Module):
    """Frozen native noise convolution + zero-initialized spatial injection."""
    def __init__(self,original,channels):
        super().__init__()
        if original.in_channels!=4:raise ValueError('Wrap the original four noise channels only once')
        self.original=original.requires_grad_(False)
        self.spatial=nn.Conv2d(channels,original.out_channels,original.kernel_size,
             original.stride,original.padding,bias=False,device=original.weight.device,dtype=torch.float32)
        nn.init.zeros_(self.spatial.weight)
        self.in_channels=4+channels
        self.out_channels=original.out_channels

    def forward(self,x):
        if x.shape[1]!=self.in_channels:raise ValueError('Spatial/noise channel contract mismatch')
        return self.original(x[:,:4].contiguous())+self.spatial(x[:,4:].to(self.spatial.weight.dtype))


class SpatialTurboHead(NativeTurboHead):
    def __init__(self,unet,vae,scheduler,adapter,spatial,positions,semantic_prefix=0):
        super().__init__(unet,vae,scheduler,adapter)
        if len(positions)!=spatial.grid**2 or len(set(positions))!=len(positions):
            raise ValueError('Invalid image token positions')
        self.spatial=spatial
        self.semantic_prefix=semantic_prefix
        self.register_buffer('visual_positions',torch.tensor(positions,dtype=torch.long),persistent=False)
        self.zero_spatial=False
        self.instruction_span = None
        self.text_readout_gain = 0.
        self.multimodal_semantic_readout = False

    def visual_tokens(self,hidden,valid):
        positions=self.visual_positions
        if positions.max()>=hidden.shape[1] or not valid[:,positions].all():
            raise ValueError('Cached Qwen token layout differs from the verified image grid')
        return hidden[:,positions]

    def forward(self,hidden,valid,noise):
        z,rgb,_=self.forward_with_aux(hidden,valid,noise,with_aux=False)
        return z,rgb

    def semantic_conditions(self,hidden,valid):
        if self.instruction_span is not None:
            from .qwen_vl_postfree import focus_features
            _, mask = focus_features(hidden, valid, self.visual_positions, *self.instruction_span, self.text_readout_gain)
            if self.multimodal_semantic_readout:
                # In native text-first order the instruction states precede
                # the image and cannot encode its contents. Let the SAME
                # existing resampler also read image and final fused states.
                mask=mask.clone();mask[:,self.visual_positions]=valid[:,self.visual_positions]
                mask.scatter_(1,(valid.sum(1)-1)[:,None],True)
            return self.adapter(hidden, mask)
        return self.adapter(hidden[:,self.semantic_prefix:],valid[:,self.semantic_prefix:])

    def spatial_inputs(self,hidden,valid):
        visual=self.visual_tokens(hidden,valid)
        context=None
        if self.instruction_span is not None:
            from .qwen_vl_postfree import focus_features
            visual, mask = focus_features(hidden, valid, self.visual_positions, *self.instruction_span, self.text_readout_gain)
            if self.spatial.film is not None:context=(hidden.float()*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        return visual,context

    def forward_with_aux(self,hidden,valid,noise,with_aux=True):
        visual,context=self.spatial_inputs(hidden,valid)
        if with_aux:spatial,aux=self.spatial(visual,with_aux=True,context=context)
        else:spatial,aux=self.spatial(visual,context=context),None
        if self.zero_spatial:spatial=spatial*0
        conditions=self.semantic_conditions(hidden,valid)
        self.scheduler.set_timesteps(1,device=noise.device)
        t=self.scheduler.timesteps[0]
        latents=noise.to(self.unet.dtype)*self.scheduler.init_noise_sigma
        noisy=self.scheduler.scale_model_input(latents,t)
        inputs=torch.cat([noisy,spatial.to(noisy.dtype)],1)
        with torch.autocast(noise.device.type,dtype=torch.bfloat16,enabled=noise.is_cuda):
            prediction=self.unet(inputs,t,encoder_hidden_states=conditions.to(self.unet.dtype)).sample
        generated=self.scheduler.step(prediction,t,latents).prev_sample
        rgb=self.vae.decode(generated.to(self.vae.dtype)/self.vae.config.scaling_factor).sample
        return generated.float(),rgb.float(),aux


def balanced_region_l1(prediction,target,edit_region):
    error=(prediction.float()-target.float()).abs()
    region=F.interpolate(edit_region.float(),size=error.shape[-2:],mode='area')
    losses=[]
    for r in (region,1-region):
        area=r.sum((1,2,3))
        losses.append(((error*r).sum((1,2,3))/(area.clamp_min(1)*error.shape[1]),area>1e-6))
    values=sum(loss*valid for loss,valid in losses)
    count=sum(valid.float() for _,valid in losses).clamp_min(1)
    return (values/count).mean()


def weighted_edit_regression(prediction,target,edit_region,edit_factor=1.,keep_factor=1.,squared=False):
    """TRAIN label regions only; absolute factors retain meaning for all-edit joint rows."""
    if min(edit_factor,keep_factor)<0 or not 0<edit_factor+keep_factor:
        raise ValueError('Nonnegative, nonzero regression factors required')
    error=prediction.float()-target.float();error=error.square() if squared else error.abs()
    region=F.interpolate(edit_region.float(),size=error.shape[-2:],mode='area')
    values=[]
    for support,factor in ((region,edit_factor),(1-region,keep_factor)):
        area=support.sum((1,2,3));valid=area>1e-6
        loss=(error*support).sum((1,2,3))/(area.clamp_min(1)*error.shape[1])
        values.append((factor*loss*valid,valid))
    return (sum(v for v,_ in values)/sum(v.float() for _,v in values).clamp_min(1)).mean()


def build_spatial_head(sd,state,cfg,positions,device,drop_vae_encoder=False,identity_probe=None):
    from diffusers import AutoencoderKL,EulerDiscreteScheduler,UNet2DConditionModel
    from peft import LoraConfig
    adapter=QueryResampler(hidden=cfg['adapter_hidden'],layers=cfg['adapter_layers']).to(device)
    adapter.load_state_dict(state['adapter'])
    unet=UNet2DConditionModel.from_pretrained(sd,subfolder='unet',variant='fp16',
        torch_dtype=torch.bfloat16,local_files_only=True).to(device).requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg['unet_lora_rank'],lora_alpha=cfg['unet_lora_rank'],
        target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights=True))
    configure_unet_training(unet,cfg['unet_training'])
    for p in unet.parameters():
        if p.requires_grad:p.data=p.data.float()
    saved=state['unet_trainable']
    expected={n for n,p in unet.named_parameters() if p.requires_grad}
    if not (set(saved)-{'conv_in.spatial.weight'}).issubset(expected):
        raise ValueError('Parent/checkpoint UNet modification scope is incompatible')
    with torch.no_grad():
        for n,p in unet.named_parameters():
            if n in saved:p.copy_(saved[n].to(device))
    vae=AutoencoderKL.from_pretrained(sd,subfolder='vae',variant='fp16',
        torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
    scheduler=EulerDiscreteScheduler.from_pretrained(sd,subfolder='scheduler',local_files_only=True)
    expected_output=None
    if identity_probe is not None:
        if 'spatial' in state:raise ValueError('Zero-injection identity applies to an unaugmented parent only')
        base=NativeTurboHead(unet,vae,scheduler,adapter).eval()
        h,valid,noise=identity_probe;prefix=cfg.get('semantic_prefix',0)
        with torch.no_grad():expected_output=base(h[:,prefix:],valid[:,prefix:],noise)
    unet.conv_in=SpatialInputConv(unet.conv_in,cfg['spatial_channels'])
    unet.register_to_config(in_channels=4+cfg['spatial_channels'])
    if 'conv_in.spatial.weight' in saved:
        with torch.no_grad():unet.conv_in.spatial.weight.copy_(saved['conv_in.spatial.weight'].to(device))
    spatial=SpatialBridge(width=cfg['spatial_width'],channels=cfg['spatial_channels'],grid=cfg['visual_grid'],conditional=cfg.get('conditional_spatial',False)).to(device)
    if cfg.get('conditional_spatial') and not cfg.get('focus_instruction'):raise ValueError('Conditional spatial readout needs verified instruction masking')
    if 'spatial' in state:
        missing,unexpected=spatial.load_state_dict(state['spatial'],strict=False)
        allowed={'film.weight','film.bias'} if cfg.get('initialize_conditional_spatial') and cfg.get('conditional_spatial') and not state['config'].get('conditional_spatial') else set()
        if unexpected or set(missing)-allowed:raise ValueError('Spatial checkpoint scope differs: '+str((missing,unexpected)))
    model=SpatialTurboHead(unet,vae,scheduler,adapter,spatial,positions,cfg.get('semantic_prefix',0))
    if cfg.get('focus_instruction'):
        model.instruction_span = tuple(cfg['instruction_bounds'])
        model.text_readout_gain = cfg.get('text_readout_gain', .15)
    if cfg.get('multimodal_semantic_readout',False):
        if not cfg.get('focus_instruction') or cfg.get('native_input_order')!='text_first_canonical_readout':
            raise ValueError('Multimodal semantic readout requires the audited focused text-first layout')
        model.multimodal_semantic_readout=True
    if identity_probe is not None:
        model.eval()
        with torch.no_grad():actual=model(*identity_probe)
        differences=[float((a-b).abs().max()) for a,b in zip(actual,expected_output)]
        model.initial_identity={'same_device':True,'latent_max_absolute_difference':differences[0],
          'rgb_max_absolute_difference':differences[1],'passed':max(differences)<1e-5}
        if not model.initial_identity['passed']:raise ValueError('Same-device zero-injection forward identity failed: '+str(model.initial_identity))
    if drop_vae_encoder:
        vae.encoder=None;vae.quant_conv=None
        spatial.aux_reference=None
    return model
