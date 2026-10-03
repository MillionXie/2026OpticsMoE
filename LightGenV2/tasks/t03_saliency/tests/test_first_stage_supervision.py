from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from experiments.vision2_hybrid_dense.modeling import SaliencyDensityDecoder, restore_qwen_block_major_spatial
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
from LightGenV2.tasks.t03_saliency.first_stage_supervision import FirstStageSupervisor, configure
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.sam_training import train_sam_epoch
from LightGenV2.tasks.t03_saliency.training import _checkpoint
from LightGenV2.tasks.t03_saliency.training_support import ModelEMA


def settings():
    s = load_settings(Path(__file__).resolve().parents[1]/'configs/moe_alpha40_first_stage_gt.yaml')
    s.map_kd_weight = 2.
    s.first_stage_current_weight = .2
    s.first_stage_head_warmup = False
    return s


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.hybrid = nn.Module()
        self.core.hybrid.blocks = nn.ModuleList([nn.Linear(192,192), nn.Linear(192,192)])
        self.core.hybrid.first_phase = nn.Parameter(torch.ones(192))
        self.core.hybrid.second_phase = nn.Parameter(torch.ones(192))
        self.head = SaliencyDensityDecoder(input_dim=192,output_size=224)
        self.calls = 0

    def forward(self, pixel_values, image_grid_thw):
        self.calls += 1
        h = self.core.hybrid
        first = h.blocks[0](pixel_values) * h.first_phase
        second = h.blocks[1](first) * h.second_phase
        spatial = restore_qwen_block_major_spatial(second.reshape(-1,192),image_grid_thw)
        return self.head(spatial),spatial,None

    def router_losses(self):
        zero = self.core.hybrid.first_phase.new_zeros(())
        return zero,zero


def batch():
    return dict(images=torch.randn(2,196,192), grid=torch.tensor([[1,14,14],[1,14,14]]),
                density=torch.rand(2,1,224,224),fixation=torch.rand(2,1,224,224)>.98,
                sample_ids=['train/a','train/b'])


def test_capture_identity_warmup_gradient_scope_and_cleanup():
    model=Toy();rng=torch.get_rng_state().clone();aux=FirstStageSupervisor(model)
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in aux.parameters())==85412
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(model.head.parameters(),aux.decoder.parameters()))
    original=deepcopy(model.state_dict());b=batch();s=settings()
    expected=model(b['images'],b['grid'])[0]
    for warm in [True,False]:
        model.zero_grad(set_to_none=True);aux.zero_grad(set_to_none=True)
        with aux.capture(model) as captured:
            actual=model(b['images'],b['grid'])[0]
        assert torch.equal(expected,actual)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            loss,cc=aux.loss(captured,b['grid'],b['density'],b['fixation'],s,detach_student=warm)
        assert loss.dtype==torch.float32 and torch.isfinite(loss+cc)
        loss.backward()
        assert all(p.grad is not None for p in aux.parameters())
        assert model.core.hybrid.second_phase.grad is None
        assert all(p.grad is None for p in model.core.hybrid.blocks[1].parameters())
        assert all(p.grad is None for p in model.head.parameters())
        assert (model.core.hybrid.first_phase.grad is None)==warm
        if not warm: assert model.core.hybrid.first_phase.grad.abs().sum()>0
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert not model.core.hybrid.blocks[1]._forward_pre_hooks
    with pytest.raises(RuntimeError,match='missing'):
        with aux.capture(model): pass
    with pytest.raises(RuntimeError,match='intentional'):
        with aux.capture(model): raise RuntimeError('intentional')
    assert not model.core.hybrid.blocks[1]._forward_pre_hooks
    with pytest.raises(ValueError,match='grids'):
        aux.loss(captured,torch.tensor([[1,7,28],[1,7,28]]),b['density'],b['fixation'],s)


def test_config_preserves_inference_and_rejects_confounders():
    s=settings();root=Path(__file__).resolve().parents[1]/'configs'
    old=load_settings(root/'moe_alpha40_extra_control.yaml')
    assert architecture_label(s)==architecture_label(old)
    for key in ['initialization_checkpoint_sha256','student_epochs','sam_rho','ema_decay','top_k',
                'fusion_alpha_min','ccd_normalization','phase_learning_rate','student_learning_rate']:
        assert getattr(s,key)==getattr(old,key)
    for key,value in [('masked_distillation',{'x':1}),('augmentation_enabled',True),('sam_rho',0),
                      ('fusion_alpha_min',.1),('top_k',1),('image_size',448)]:
        bad=deepcopy(s);setattr(bad,key,value)
        with pytest.raises(ValueError):configure(bad,s.first_stage_supervision)
    for key,value in [('initial_weight',float('nan')),('head_warmup_epochs',True),('end_epoch',2)]:
        opts=dict(s.first_stage_supervision);opts[key]=value
        with pytest.raises(ValueError):configure(deepcopy(s),opts)


def test_sam_two_forwards_one_update_and_last_only_aux(tmp_path,monkeypatch):
    model=Toy();aux=FirstStageSupervisor(model);s=settings();b=batch()
    s.phase_dc_weight=s.router_balance_weight=s.router_importance_weight=s.ccd_operating_point_weight=0.
    opt=torch.optim.AdamW([{'params':model.core.parameters(),'name':'electronic'},
        {'params':model.head.parameters(),'name':'saliency_head'},
        {'params':aux.parameters(),'name':'training_first_stage'}],lr=.001)
    ema=ModelEMA(model,.9);steps=[]
    eh=opt.register_step_post_hook(lambda *args:(steps.append(1),ema.update(*args)))
    monkeypatch.setattr(legacy,'_autocast',lambda *a:nullcontext())
    monkeypatch.setattr(legacy,'preprocess_vision',lambda *a:dict(pixel_values=b['images'],image_grid_thw=b['grid']))
    before=deepcopy(aux.state_dict());seen=[]
    hook=aux.decoder.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
    result=train_sam_epoch(model,[b],SimpleNamespace(device=torch.device('cpu'),processor=None),s,opt,first_stage=aux)
    hook.remove();eh.remove()
    assert model.calls==2 and len(steps)==1 and len(seen)==2
    assert result['samples']==2 and result['first_stage_loss']>0
    assert any(not torch.equal(v,aux.state_dict()[k]) for k,v in before.items())
    assert not model.core.hybrid.blocks[1]._forward_pre_hooks
    model.core.hybrid.fusion_alpha_min=.4;model.core.hybrid.fusion_alpha_max=.95
    model.checkpoint_architecture='toy'
    _checkpoint(tmp_path/'last.pt',model,1,{},None,training_only_first_stage=aux.state_dict())
    _checkpoint(tmp_path/'best.pt',model,1,{}, {'cc':.1})
    assert 'training_only_first_stage' in torch.load(tmp_path/'last.pt',weights_only=False)
    assert 'training_only_first_stage' not in torch.load(tmp_path/'best.pt',weights_only=False)
