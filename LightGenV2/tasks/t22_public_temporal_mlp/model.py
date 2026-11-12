"""Explicit serial surrogate. No imports from the private/paper model.

Only the two global stages propagate light. The four old electronic routes,
RMS fusions, quality14, temporal convolutions/statistical head are absent.
All locations previously using Softplus use ReLU in this model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


GELU_ARCHITECTURE = 't22_serial_mlp_top2_global2_relu_original_merger_separate_inputs_fc2'
RELU_ARCHITECTURE = GELU_ARCHITECTURE + '_allrelu'
FIVE_LEVEL_ARCHITECTURE = RELU_ARCHITECTURE + '_five_levels'
AMPLITUDE_ARCHITECTURE = 't22_serial_top2_global2_five_levels_amplitude_v6'
JOINT_ROUTER_ARCHITECTURE = 't22_serial_top2_global2_five_levels_amplitude_v7_joint_router'
DIRECT_PHASE_ARCHITECTURE = 't22_serial_top2_global2_five_levels_amplitude_v8_direct_phase_joint_router'
QUALITY_WORDS = ('Bad', 'Poor', 'Fair', 'Good', 'Excellent')


@dataclass(frozen=True)
class ModelConfig:
    width: int = 192
    expert_hidden: int = 384
    router_hidden: int = 128
    fusion_router_input: str = 'visual'
    readout_hidden: int = 1024
    expert_mode: str = 'mlp'
    orm_mode: str = 'legacy_energy'
    alpha_min: float = 2 / 3
    alpha_max: float = 2 / 3
    alpha_eval: float = 2 / 3
    mlp_activation: str = 'relu'
    readout_mode: str = 'five_levels'
    visual_output_norm: bool = False
    prompt_length: int = 38
    videos: int = 16
    frames: int = 4
    tokens: int = 49
    temperature: float = 1.10
    router_noise: float = 0.06
    input_shift: int = 4
    phase_shift: int = 4
    ccd_shift: int = 4
    phase_dropout: float = 0.05
    phase_dropout_cell: int = 4
    phase_noise_std: float = 0.0
    leakage_min: float = 0.20
    leakage_max: float = 0.35
    leakage_eval: float = 0.20
    wavelength_nm: float = 532.0
    pitch_um: float = 17.0
    distance_m: float = 0.10
    cutoff_deg: float = 0.50
    canvas: int = 518
    active: int = 478
    phase_parameterization: str = 'bounded'

    def __post_init__(self):
        if (self.videos,self.frames,self.tokens,self.canvas,self.active)!=(16,4,49,518,478):
            raise ValueError('This contract fixes 16 videos x 4 frames, 7x7 tokens, 518/478 optics')
        if not 0 < self.prompt_length <= 52:
            raise ValueError('4 frame tokens plus the fixed prompt must fit the 56-row video field')
        if self.width <= 0 or self.temperature <= 0:
            raise ValueError('Invalid model width/temperature')
        if self.expert_mode not in ('mlp', 'linear'):
            raise ValueError('Unsupported expert mode')
        if self.fusion_router_input not in ('visual','all_tokens'):
            raise ValueError('Unsupported fusion router input')
        if self.fusion_router_input == 'all_tokens' and self.orm_mode != 'amplitude':
            raise ValueError('The joint router requires the current amplitude architecture')
        if self.phase_parameterization not in ('bounded', 'direct'):
            raise ValueError('Unsupported phase parameterization')
        if self.phase_parameterization == 'direct' and self.fusion_router_input != 'all_tokens':
            raise ValueError('Direct phase requires the joint image-text router architecture')
        if not math.isfinite(self.phase_noise_std) or self.phase_noise_std < 0:
            raise ValueError('Phase noise standard deviation must be finite and nonnegative')
        if self.orm_mode not in ('legacy_energy', 'amplitude'):
            raise ValueError('Unsupported ORM mode')
        if not 0 <= self.alpha_min <= self.alpha_max <= 1 or not 0 <= self.alpha_eval <= 1:
            raise ValueError('ORM amplitude weights must lie in [0, 1]')
        if self.expert_mode == 'linear' and self.orm_mode != 'amplitude':
            raise ValueError('The linear-expert ablation requires amplitude ORM')
        if self.orm_mode == 'amplitude' and self.readout_mode != 'five_levels':
            raise ValueError('Amplitude ORM requires the current five-level readout')
        if self.mlp_activation not in ('relu', 'gelu'):
            raise ValueError('Unsupported MLP activation')
        if self.readout_mode not in ('scalar', 'five_levels'):
            raise ValueError('Unsupported readout mode')
        if self.readout_mode == 'five_levels' and self.mlp_activation != 'relu':
            raise ValueError('The five-level architecture uses ReLU')
        if self.readout_mode == 'scalar' and not self.visual_output_norm:
            raise ValueError('Historical scalar profiles require the visual output LayerNorm')


def mlp_activation(config: ModelConfig) -> nn.Module:
    if config.mlp_activation == 'relu':
        return nn.ReLU()
    return nn.GELU()


def translated(value, dy, dx, fill=0.0):
    out=torch.roll(value,(dy,dx),dims=(-2,-1))
    if dy>0: out[..., :dy,:]=fill
    elif dy<0: out[..., dy:,:]=fill
    if dx>0: out[..., :, :dx]=fill
    elif dx<0: out[..., :, dx:]=fill
    return out


def random_shift(maximum, training):
    return tuple(torch.randint(-maximum,maximum+1,(2,)).tolist()) if training and maximum else (0,0)


class FrozenFront(nn.Module):
    """Extracted Qwen patch+position only; no Qwen model/tokenizer at runtime.

Input is already cropped/resized uint8 RGB [...,3,448,448]. Each image is
duplicated into the two temporal slices expected by its frozen patch kernel.
"""
    def __init__(self):
        super().__init__()
        self.register_buffer('patch_weight',torch.zeros(1024,3,2,16,16,dtype=torch.bfloat16))
        self.register_buffer('patch_bias',torch.zeros(1024,dtype=torch.bfloat16))
        self.register_buffer('position',torch.zeros(784,1024,dtype=torch.bfloat16))
        self.register_buffer('ready',torch.tensor(False))

    def forward(self, images):
        if not bool(self.ready):
            raise RuntimeError('Extracted front weights have not been populated')
        if tuple(images.shape[-3:]) != (3,448,448) or images.dtype!=torch.uint8:
            raise ValueError('Frozen front requires uint8 [...,3,448,448] RGB')
        shape=images.shape[:-3]
        rgb=(images.reshape(-1,3,448,448).float()/255.0-0.5)/0.5
        rgb=rgb.unsqueeze(2).expand(-1,-1,2,-1,-1).to(self.patch_weight.dtype)
        hidden=F.conv3d(rgb,self.patch_weight,self.patch_bias,stride=(2,16,16))[:,:,0]
        # Preserve the official processor's contiguous 2x2 merge-group order.
        hidden=hidden.reshape(-1,1024,14,2,14,2).permute(0,2,4,3,5,1).reshape(-1,784,1024)
        hidden=hidden+self.position
        merged=hidden.reshape(-1,196,4,1024).mean(2).reshape(-1,14,14,1024)
        pooled=merged.reshape(-1,7,2,7,2,1024).mean((2,4))
        # Match the historical fp16 feature cache boundary.
        return pooled.reshape(*shape,49,1024).to(torch.float16)


class Router(nn.Module):
    def __init__(self,cfg:ModelConfig,token_count:int):
        super().__init__();self.cfg=cfg
        self.norm=nn.LayerNorm(cfg.width)
        self.fc1=nn.Linear(token_count*cfg.width,cfg.router_hidden)
        self.fc2=nn.Linear(cfg.router_hidden,4)
        self.activation = mlp_activation(cfg)

    def forward(self,tokens):
        logits=self.fc2(self.activation(self.fc1(self.norm(tokens).flatten(-2))))
        if self.training and self.cfg.router_noise:
            logits=logits+self.cfg.router_noise*torch.randn_like(logits)
        probabilities=(logits/self.cfg.temperature).softmax(-1)
        indices=probabilities.topk(2,dim=-1).indices
        selected=torch.zeros_like(probabilities).scatter(-1,indices,1.0)
        weights=probabilities*selected
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return {'weights':weights,'probabilities':probabilities,'selected':selected.bool()}


class Experts(nn.Module):
    """Four shared tokenwise experts, with one explicitly selected definition."""
    def __init__(self,cfg):
        super().__init__()
        if cfg.expert_mode == 'linear':
            self.experts = nn.ModuleList([
                nn.Linear(cfg.width, cfg.width, bias=False) for _ in range(4)])
        else:
            self.experts=nn.ModuleList([nn.Sequential(nn.LayerNorm(cfg.width),
                nn.Linear(cfg.width,cfg.expert_hidden),mlp_activation(cfg),
                nn.Linear(cfg.expert_hidden,cfg.width)) for _ in range(4)])

    def forward(self,tokens,weights):
        # All four are evaluated for a simple implementation; exactly two
        # contribute per decision in the forward pass. This is not a sparse
        # hardware-speed claim.
        out=torch.zeros_like(tokens)
        for i,expert in enumerate(self.experts):
            out=out+expert(tokens)*weights[...,i,None,None]
        return out


class Propagation(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        f=torch.fft.fftfreq(cfg.canvas,d=cfg.pitch_um*1e-6,dtype=torch.float64)
        fy,fx=torch.meshgrid(f,f,indexing='ij');wavelength=cfg.wavelength_nm*1e-9
        argument=(2*math.pi)**2*((1/wavelength)**2-fx.square()-fy.square())
        radial=2*math.pi*torch.sqrt(fx.square()+fy.square())
        valid=(argument>=0)&(radial<=(2*math.pi/wavelength)*math.sin(math.radians(cfg.cutoff_deg)))
        transfer=torch.exp(1j*cfg.distance_m*torch.sqrt(argument.clamp_min(0))).to(torch.complex64)
        self.register_buffer('transfer',torch.where(valid,transfer,torch.zeros_like(transfer)),persistent=False)

    def forward(self,field):
        return torch.fft.ifft2(torch.fft.fft2(field.to(torch.complex64))*self.transfer)


class GlobalStage(nn.Module):
    """Original global layout/physics, with explicitly requested ReLU activations."""
    def __init__(self,cfg,frame_level):
        super().__init__();self.cfg=cfg;self.frame_level=frame_level
        side=27 if frame_level else 56
        self.width_to_field=nn.Linear(cfg.width,side)
        if frame_level:
            self.tokens_to_field=nn.Linear(cfg.tokens,side)
        self.raw_global_phase=nn.Parameter(torch.empty(16,111,111))
        if cfg.phase_parameterization == 'direct':
            nn.init.zeros_(self.raw_global_phase)
        else:
            nn.init.normal_(self.raw_global_phase,0,0.25)
        self.propagation=Propagation(cfg)
        self.norm=nn.LayerNorm(96);self.output=nn.Linear(96,cfg.width)
        support=torch.zeros(518,518,dtype=torch.bool)
        for y,x in self.origins:
            support[y:y+115,x:x+115]=True
        self.register_buffer('support',support,persistent=False)

    @property
    def origins(self):
        return [(20+3+119*y,20+3+119*x) for y in range(4) for x in range(4)]

    def fields(self,tokens):
        value=F.relu(self.width_to_field(tokens))
        if self.frame_level:
            value=F.relu(self.tokens_to_field(value.transpose(-2,-1))).transpose(-2,-1)
        else:
            value=F.pad(value,(0,0,0,56-value.shape[-2]))
        # ReLU can produce an exactly dark field. Clamp before sqrt so its
        # backward pass cannot encounter 0 * infinity at that point.
        return value/value.square().mean((-2,-1),keepdim=True).clamp_min(1e-12).sqrt()

    def phase(self, *, wrapped=False):
        if self.cfg.phase_parameterization == 'direct':
            value = self.raw_global_phase
        else:
            value = 2 * math.pi * torch.sigmoid(self.raw_global_phase)
        return value.remainder(2 * math.pi) if wrapped else value

    def modulation(self):
        phase=self.phase()
        if self.training and self.cfg.phase_noise_std:
            phase = phase + self.cfg.phase_noise_std*torch.randn_like(phase)
        value=torch.exp(1j*phase)
        if self.training and self.cfg.phase_dropout:
            k=self.cfg.phase_dropout_cell
            keep=(torch.rand(16,math.ceil(111/k),math.ceil(111/k),device=value.device)>=self.cfg.phase_dropout)
            keep=keep.repeat_interleave(k,-2).repeat_interleave(k,-1)[...,:111,:111]
            value=torch.where(keep,value,torch.ones_like(value))
        if self.cfg.orm_mode == 'amplitude':
            alpha = (self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min)
                     * torch.rand((), device=value.device) if self.training
                     else value.real.new_tensor(self.cfg.alpha_eval))
            # Manuscript V = alpha * D U + (1-alpha) * U_tilde.
            # Here U_tilde = U, so the unmodulated residual has unit phase.
            value = alpha * value + (1 - alpha)
        else:
            eta=(self.cfg.leakage_min+(self.cfg.leakage_max-self.cfg.leakage_min)*torch.rand((),device=value.device)
                 if self.training else value.real.new_tensor(self.cfg.leakage_eval))
            value=(1-eta).sqrt()*value+eta.sqrt()
        return translated(value,*random_shift(self.cfg.phase_shift,self.training),fill=1+0j)

    def forward(self,tokens):
        fields=translated(self.fields(tokens),*random_shift(self.cfg.input_shift,self.training))
        canvas=fields.new_zeros(len(fields),518,518)
        for v,(y,x) in enumerate(self.origins):
            if self.frame_level:
                for frame,(dy,dx) in enumerate([(0,0),(0,59),(59,0),(59,59)]):
                    canvas[:,y+dy+14:y+dy+41,x+dx+14:x+dx+41]=fields[:,v,frame]
            else:
                canvas[:,y+29:y+85,x+29:x+85]=fields[:,v]
        # One full coherent canvas, not 16 independently propagated tiles.
        phase=self.modulation();mask=torch.ones_like(canvas,dtype=torch.complex64)
        for v,(y,x) in enumerate(self.origins):
            mask[:,y+2:y+113,x+2:x+113]=phase[v]
        detector=self.propagation(canvas.to(torch.complex64)*mask).abs().square().float()
        detector=translated(detector,*random_shift(self.cfg.ccd_shift,self.training))
        guard=(detector.masked_fill(self.support,0).sum((-2,-1))/detector.sum((-2,-1)).clamp_min(1e-8)).mean()
        patches=[]
        for y,x in self.origins:
            if self.frame_level:
                patches.append(torch.stack([detector[:,y+dy:y+dy+56,x+dx:x+dx+56]
                    for dy,dx in [(0,0),(0,59),(59,0),(59,59)]],1))
            else: patches.append(detector[:,y:y+115,x:x+115])
        patches=torch.stack(patches,1).clamp_min(0)
        patches=torch.log1p((patches/patches.mean((-2,-1),keepdim=True).clamp_min(1e-6)).clamp_max(8))
        leading=patches.shape[:-2];count=self.cfg.tokens if self.frame_level else tokens.shape[-2]
        pooled=F.adaptive_avg_pool2d(patches.reshape(-1,1,*patches.shape[-2:]),(count,96))[:,0]
        output=self.output(F.relu(self.norm(pooled)))
        return output.reshape(*leading,count,self.cfg.width),guard


class LightGenTemporal(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__();self.cfg=cfg
        self.front=FrozenFront()
        self.register_buffer('prompt_embedding',torch.zeros(cfg.prompt_length,2048,dtype=torch.float16))
        self.register_buffer('prompt_ready',torch.tensor(False))
        self.vision_adapter=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,cfg.width))
        self.language_adapter=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,cfg.width))
        self.visual_input_norm = (
            nn.LayerNorm(cfg.width) if cfg.visual_output_norm else nn.Identity())
        self.frame_router=Router(cfg,49);self.frame_experts=Experts(cfg)
        self.frame_global=GlobalStage(cfg,True)
        self.frame_merger=nn.Sequential(nn.LayerNorm(2*cfg.width),
            nn.Linear(2*cfg.width,cfg.width),mlp_activation(cfg))
        self.frame_position=nn.Parameter(torch.zeros(1,1,4,cfg.width))
        self.sequence_position=nn.Parameter(torch.zeros(1,64,cfg.width))
        nn.init.normal_(self.frame_position,std=.02);nn.init.normal_(self.sequence_position,std=.02)
        router_tokens = 4+cfg.prompt_length if cfg.fusion_router_input == 'all_tokens' else 4
        self.video_router=Router(cfg,router_tokens);self.video_experts=Experts(cfg)
        self.video_global=GlobalStage(cfg,False)
        self.readout=nn.Sequential(nn.Flatten(-2),nn.Linear((4+cfg.prompt_length)*cfg.width,cfg.readout_hidden),
            mlp_activation(cfg),nn.Linear(cfg.readout_hidden,5 if cfg.readout_mode == 'five_levels' else 1))
        self.register_buffer('target_mean',torch.tensor(0.))
        self.register_buffer('target_std',torch.tensor(1.))
        if cfg.readout_mode == 'five_levels':
            self.register_buffer('level_scores', torch.zeros(5))
            self.register_buffer('level_boundaries', torch.zeros(4))
            self.register_buffer('quality_ready', torch.tensor(False))

    def set_quality_levels(self, train_targets):
        if self.cfg.readout_mode != 'five_levels':
            raise ValueError('Quality levels require the five-level readout')
        targets = torch.as_tensor(train_targets, dtype=torch.float32)
        if not bool(torch.isfinite(targets).all()) or targets.numel() < 2:
            raise ValueError('Finite TRAIN MOS targets are required')
        lower, upper = float(targets.min()), float(targets.max())
        if lower >= upper:
            raise ValueError('TRAIN MOS range must be positive')
        self.level_scores.copy_(torch.linspace(lower, upper, 5).to(self.level_scores))
        self.level_boundaries.copy_(torch.linspace(lower, upper, 6)[1:-1].to(self.level_boundaries))
        self.quality_ready.fill_(True)

    def set_prompt(self,embedding):
        value=embedding.reshape(-1,2048)
        if value.shape!=self.prompt_embedding.shape:raise ValueError('Fixed prompt shape mismatch')
        self.prompt_embedding.copy_(value.to(self.prompt_embedding));self.prompt_ready.fill_(True)

    def encode_frames(self,vision_tokens):
        """Visual stage through the original merger, using only image features."""
        if tuple(vision_tokens.shape[1:])!=(16,4,49,1024):raise ValueError('Expected [B,16,4,49,1024]')
        vision=self.visual_input_norm(self.vision_adapter(vision_tokens.float()))
        r1=self.frame_router(vision)
        vision=self.frame_experts(vision,r1['weights'])
        vision,g1=self.frame_global(vision)
        frame_summary=torch.cat((vision.mean(-2),vision.amax(-2)),-1)
        frames=self.frame_merger(frame_summary)+self.frame_position
        return {'frames':frames,'routing':r1,'guard_energy':g1}

    def forward(self,vision_tokens=None,*,pixels=None):
        if (vision_tokens is None)==(pixels is None):raise ValueError('Provide exactly one of tokens/pixels')
        if pixels is not None:vision_tokens=self.front(pixels)
        if not bool(self.prompt_ready):raise RuntimeError('Fixed prompt embedding is not populated')
        visual=self.encode_frames(vision_tokens)
        frames=visual['frames']
        prompt=self.language_adapter(self.prompt_embedding.float())
        prompt=prompt[None,None].expand(len(frames),16,-1,-1)
        sequence=torch.cat((frames,prompt),dim=-2)
        sequence=sequence+self.sequence_position[:,None,:sequence.shape[-2]]
        router_input = sequence if self.cfg.fusion_router_input == 'all_tokens' else sequence[...,:4,:]
        r2=self.video_router(router_input)
        sequence=self.video_experts(sequence,r2['weights'])
        sequence,g2=self.video_global(sequence)
        output = self.readout(sequence)
        result = {'routing': {'frame': visual['routing'], 'video': r2},
                  'guard_energy': (visual['guard_energy'] + g2) / 2}
        if self.cfg.readout_mode == 'five_levels':
            if not bool(self.quality_ready):
                raise RuntimeError('TRAIN quality levels have not been populated')
            probabilities = output.softmax(-1)
            prediction = probabilities @ self.level_scores
            result.update(prediction=prediction, quality_logits=output,
                          quality_probabilities=probabilities,
                          normalized_prediction=(prediction - self.target_mean) / self.target_std)
        else:
            normalized = output.squeeze(-1)
            result.update(prediction=normalized * self.target_std + self.target_mean,
                          normalized_prediction=normalized)
        return result

    def checkpoint(self,**metadata:Any):
        all_relu = self.cfg.mlp_activation == 'relu'
        architecture = RELU_ARCHITECTURE if all_relu else GELU_ARCHITECTURE
        schema = 4 if all_relu else 3
        if self.cfg.readout_mode == 'five_levels':
            suffix = '_ln192' if self.cfg.visual_output_norm else '_no_ln192'
            architecture = FIVE_LEVEL_ARCHITECTURE + suffix
            schema = 5
        if self.cfg.orm_mode == 'amplitude':
            suffix = '_ln192' if self.cfg.visual_output_norm else '_no_ln192'
            architecture = AMPLITUDE_ARCHITECTURE + '_' + self.cfg.expert_mode + suffix
            schema = 6
            if self.cfg.fusion_router_input == 'all_tokens':
                architecture = JOINT_ROUTER_ARCHITECTURE + '_' + self.cfg.expert_mode + suffix
                schema = 7
                if self.cfg.phase_parameterization == 'direct':
                    architecture = DIRECT_PHASE_ARCHITECTURE + '_' + self.cfg.expert_mode + suffix
                    schema = 8
        return {'schema':schema, 'architecture':architecture,
                'config':asdict(self.cfg),'model':self.state_dict(),'metadata':metadata}


def load_lightgen(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    architecture = payload['architecture']
    activations = {GELU_ARCHITECTURE: 'gelu', RELU_ARCHITECTURE: 'relu',
                   FIVE_LEVEL_ARCHITECTURE + '_ln192': 'relu',
                   FIVE_LEVEL_ARCHITECTURE + '_no_ln192': 'relu'}
    amplitude_prefixes = (AMPLITUDE_ARCHITECTURE, JOINT_ROUTER_ARCHITECTURE, DIRECT_PHASE_ARCHITECTURE)
    for prefix in amplitude_prefixes:
        for expert in ('mlp', 'linear'):
            for norm in ('_ln192', '_no_ln192'):
                activations[prefix + '_' + expert + norm] = 'relu'
    if architecture not in activations:
        raise ValueError('Checkpoint uses a different T22 prototype architecture. Use its recorded Git version.')
    config = dict(payload['config'])
    if architecture in (GELU_ARCHITECTURE, RELU_ARCHITECTURE):
        config.setdefault('readout_mode', 'scalar')
        config.setdefault('visual_output_norm', True)
        if config['readout_mode'] != 'scalar' or not config['visual_output_norm']:
            raise ValueError('Historical scalar architecture conflicts with its configuration')
    elif architecture.startswith(tuple(prefix + '_' for prefix in amplitude_prefixes)):
        expected_norm = not architecture.endswith('_no_ln192')
        expected_expert = 'linear' if '_linear_' in architecture else 'mlp'
        if (config.get('readout_mode') != 'five_levels'
                or config.get('visual_output_norm') != expected_norm
                or config.get('expert_mode') != expected_expert
                or config.get('orm_mode') != 'amplitude'):
            raise ValueError('Amplitude architecture conflicts with its configuration')
        expected_input = ('all_tokens' if architecture.startswith(
            (JOINT_ROUTER_ARCHITECTURE + '_', DIRECT_PHASE_ARCHITECTURE + '_')) else 'visual')
        if config.get('fusion_router_input','visual') != expected_input:
            raise ValueError('Fusion router input conflicts with its architecture identity')
    else:
        expected_norm = architecture == FIVE_LEVEL_ARCHITECTURE + '_ln192'
        if config.get('readout_mode') != 'five_levels' or config.get('visual_output_norm') != expected_norm:
            raise ValueError('Five-level architecture conflicts with its configuration')
    expected_phase = 'direct' if architecture.startswith(DIRECT_PHASE_ARCHITECTURE + '_') else 'bounded'
    config.setdefault('phase_parameterization', expected_phase)
    if config['phase_parameterization'] != expected_phase:
        raise ValueError('Phase parameterization conflicts with its architecture identity')
    if not architecture.startswith(tuple(prefix + '_' for prefix in amplitude_prefixes)):
        config.setdefault('expert_mode', 'mlp')
        config.setdefault('orm_mode', 'legacy_energy')
        if config['expert_mode'] != 'mlp' or config['orm_mode'] != 'legacy_energy':
            raise ValueError('Historical architecture conflicts with expert/ORM configuration')
    config.setdefault('mlp_activation', activations[architecture])
    if config['mlp_activation'] != activations[architecture]:
        raise ValueError('Checkpoint activation conflicts with its architecture identity')
    model=LightGenTemporal(ModelConfig(**config))
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload['metadata']
