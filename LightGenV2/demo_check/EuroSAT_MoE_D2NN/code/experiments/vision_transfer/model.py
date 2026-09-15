import hashlib
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from .vision import build_classification_student, classification_logits
from .layout import A_EXPERTS, B_EXPERTS, check_layout
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import OpticalDetectorTopKRouter, sparsify_probabilities
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import MoE4LanguageTwoBlockOpticalPath
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import HomogeneousMoEOpticalCore, GlobalPhasePlane


def dense_weights(probabilities):
    p=probabilities.float()
    if p.ndim!=2 or p.shape[1]!=4 or not bool(torch.isfinite(p).all()) or bool((p<=0).any()):
        raise RuntimeError('Dense router requires four finite positive probabilities')
    return p/p.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)


class ReservedRouter(OpticalDetectorTopKRouter):
    forced_domains=None
    suppress_b=False

    def forward(self,fields):
        if self.forced_domains is not None:
            raise RuntimeError('Vision-only dense routing forbids task masks, including training')
        route=super().forward(fields)
        p=route['probabilities'];weights=dense_weights(p)
        route.update(weights=weights,selected_mask=torch.ones_like(p,dtype=torch.bool),
                     selected_indices=torch.arange(4,device=p.device).expand(len(p),-1),
                     straight_through=False,routing_mode='dense_all_four')
        if self.suppress_b:
            # Explicit validation-only ablation, never normal training/inference.
            route['weights']=weights*p.new_tensor([1,1,0,0])
        return route

class SingleChannelCore(HomogeneousMoEOpticalCore):
    def _direct_amplitude_load(self, fields, routing):
        amplitude = self._normalize_amplitude_slm_input(fields.float())
        full = F.interpolate(amplitude[:, None], size=(self.geometry.active_size,)*2,
                             mode='bilinear', align_corners=False)[:, 0]
        power = amplitude.square().sum((-2,-1), keepdim=True)
        full = full * (power / full.square().sum((-2,-1), keepdim=True).clamp_min(1e-12)).sqrt()
        a = self.geometry.active_aperture
        canvas = F.pad(full, (a.x0, self.geometry.canvas_size-a.x1, a.y0, self.geometry.canvas_size-a.y1))
        self.last_power_relative_error = ((canvas.square().sum((-2,-1))-power.flatten()).abs()/power.flatten().clamp_min(1e-12)).detach()
        return torch.complex(canvas, torch.zeros_like(canvas))

    def begin(self, fields):
        # No router invocation, expert replication, conditional mask, or top-k.
        self.last_routing = {}
        return self._direct_amplitude_load(fields, {}), {}

class SingleChannelPath(MoE4LanguageTwoBlockOpticalPath):
    def _expert_phase_modulation(self, field):
        return self.core.expert_layers[0](torch.ones_like(field, dtype=torch.complex64))

def surrogates(r):
    return [('vision', r.vision_surrogate)]

def modules(r, h):
    return [('vision_optical', r.vision_surrogate), ('classification_head', h)]

def named(r, h):
    return {prefix+'.'+n:p for prefix,m in modules(r,h) for n,p in m.named_parameters()}

def build(loaded, s, architecture):
    r,h = build_classification_student(loaded,s)
    for _,sur in surrogates(r):
        path = sur.core.optical_branch
        core = path.core
        # Vision-only keeps the same optical architecture and hardware perturbations off.
        path.input_shift_pixels=path.phase_shift_pixels=path.ccd_shift_pixels=0
        path.gain_min=path.gain_max=1.0
        path.offset_fraction=path.read_noise_fraction=0.0
        path.ccd_noise_distribution='gaussian'
        path.zero_order_enabled=False
        if architecture == 'moe':
            check_layout(core.geometry, core.router)
            core.router.__class__ = ReservedRouter
            core.router.top_k=4
            core.router.straight_through=False
            core.router.set_noise_std(0)
            core.router.input_shift_pixels=core.router.phase_shift_pixels=core.router.ccd_shift_pixels=0
            core.router.phase_dropout_p=0.0
        elif architecture == 'd2nn':
            core.__class__ = SingleChannelCore
            path.__class__ = SingleChannelPath
            core.router = nn.Identity()
            # Build the same electronics first, then replace only optical pieces.
            # A dedicated RNG keeps electronic initialization identical to MoE.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2042 + (0 if sur is r.vision_surrogate else 1))
                core.expert_layers = nn.ModuleList([GlobalPhasePlane(core.geometry,s)]).to(loaded.device)
        else:
            raise ValueError(architecture)
    r.set_phase_dropout_active(False)
    r.transfer_architecture=architecture
    r.transfer_eligible={n for n,p in named(r,h).items() if p.requires_grad}
    return r,h

def group_of(name):
    if name.endswith('raw_router_phase'): return 'router'
    if name.endswith('raw_phase'):
        match = re.search(r'\.experts\.(\d+)\.', name)
        if match: return 'expert_a' if int(match[1]) in A_EXPERTS else 'expert_b'
        return 'shared_phase'
    return 'head' if name.startswith('classification_head.') else 'electronic'

def schedule(task, epoch, architecture, variant):
    if task=='A':
        if epoch<=5: lrs=dict(electronic=2e-4,head=1e-3,expert_a=1e-3,shared_phase=1e-3,router=1e-3)
        elif epoch<=40:
            scale=.25+.75*.5*(1+math.cos(math.pi*(epoch-6)/34))
            lrs=dict(expert_a=3e-3*scale,shared_phase=1e-3*scale,router=1e-3*scale)
        else:
            scale=.1+.9*.5*(1+math.cos(math.pi*(epoch-41)/19))
            lrs=dict(electronic=5e-5*scale,head=1e-4*scale,expert_a=5e-4*scale,shared_phase=5e-4*scale,router=5e-4*scale)
    else:
        if epoch<=10: p,rt=3e-3,1e-3
        elif epoch<=30:
            t=(epoch-11)/19; p,rt=3e-3*(1-t)+1e-3*t,1e-3*(1-t)+3e-4*t
        else:
            scale=.1+.9*.5*(1+math.cos(math.pi*(epoch-31)/9))
            p,rt=1e-3*scale,3e-4*scale
        lrs=dict(expert_b=p,router=rt)
        if variant=='all': lrs['expert_a']=p
        if architecture=='d2nn': lrs={'shared_phase':p}
    return lrs

def optimizer_for(r,h,task,epoch,variant,old=None):
    rates=schedule(task,epoch,r.transfer_architecture,variant)
    groups={}
    for name,p in named(r,h).items():
        group=group_of(name); p.grad=None; p.requires_grad_(name in r.transfer_eligible and rates.get(group,0)>0)
        if p.requires_grad: groups.setdefault(group,[]).append(p)
    signature=tuple(sorted(groups))
    if old is not None and tuple(sorted(g['group_name'] for g in old.param_groups))==signature:
        for g in old.param_groups:g['lr']=rates[g['group_name']]
        return old
    # Rebuild only at changes of trainability; no momentum/decay in frozen groups.
    return torch.optim.AdamW([dict(params=ps,group_name=g,lr=rates[g],weight_decay=0.) for g,ps in groups.items()])

def set_mode(loaded,r,h,training,task=None,epoch=1):
    loaded.model.eval()
    # eval() does not stop phase gradients and locks all electronic dropout/buffers.
    for _,m in modules(r,h):m.eval()
    if training and task=='A' and (epoch<=5 or epoch>=41):
        for _,m in modules(r,h):m.train()
    if r.transfer_architecture=='moe':
        for _,sur in surrogates(r):
            sur.core.optical_branch.core.router.train(training)
            sur.core.optical_branch.core.router.forced_domains=None
    r.set_phase_dropout_active(False)

def force_route(r,domains):
    if r.transfer_architecture=='moe':
        for _,sur in surrogates(r):sur.core.optical_branch.core.router.forced_domains=domains

def routes(r):
    return {mod:sur.core.last_routing for mod,sur in surrogates(r)} if r.transfer_architecture=='moe' else {}

def clone(r,h):
    return dict(backbone_metadata=r.backbone_metadata,**{name:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for name,m in modules(r,h)})

def restore(r,h,state):
    if state.get('backbone_metadata',{}).get('frozen_stem_sha256')!=r.backbone_metadata['frozen_stem_sha256']:
        raise RuntimeError('Frozen visual stem differs from checkpoint; language checkpoints are incompatible')
    for n,m in modules(r,h):m.load_state_dict(state[n],strict=True)

def digest(r,h,frozen_only=False):
    sha=hashlib.sha256(); ps=named(r,h)
    for pref,m in modules(r,h):
        # Include nonpersistent buffers too: these can affect inference.
        values={**dict(m.named_parameters()), **dict(m.named_buffers())}
        for k,v in sorted(values.items()):
            full=pref+'.'+k
            if frozen_only and full in ps and ps[full].requires_grad:continue
            sha.update(full.encode());sha.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()

def electronic_digest(r,h):
    sha=hashlib.sha256()
    for n,p in sorted(named(r,h).items()):
        if group_of(n) in ('electronic','head'):
            sha.update(n.encode());sha.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()

def resource_report(r,h):
    counts={}
    for n,p in named(r,h).items():counts[group_of(n)]=counts.get(group_of(n),0)+p.numel()
    from .layout import spatial_table
    return dict(architecture=r.transfer_architecture,modalities=['vision'],expert_layout='upper A=(0,1), lower B=(2,3)',parameter_counts=counts,
                backbone=r.backbone_metadata,head_parameters=sum(p.numel() for p in h.parameters()),
                spatial_mapping=spatial_table(r.vision_surrogate.core.optical_branch.core.geometry,r.vision_surrogate.core.optical_branch.core.router) if r.transfer_architecture=='moe' else None,
                optical_phase_parameters=sum(v for k,v in counts.items() if k not in ('electronic','head')),
                feature_ccd_exposures=2,router_exposures=1 if r.transfer_architecture=='moe' else 0,
                final_inference='one checkpoint, fixed optical sequence, no task ID or task mask; '+('all four experts' if r.transfer_architecture=='moe' else 'one contiguous aperture, no router'),
                active_expert_channels=4 if r.transfer_architecture=='moe' else 1,
                head='LayerNorm(384) -> Linear(384,10)',electronics_initial_sha256=electronic_digest(r,h))

def predict(loaded,r,h,inputs):
    set_mode(loaded,r,h,False)
    force_route(r,None)
    return classification_logits(loaded.model,r,h,inputs)[0]
