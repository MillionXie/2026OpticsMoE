from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from LightGenV2.tasks.t03_saliency.feature_hints import TrainFeatureHints,train_hint_epoch,feature_hint_loss
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy


def cache(tmp_path):
    s=SimpleNamespace(augmentation_enabled=False,feature_hint_cache=tmp_path/'features.pt',
                      distillation_teacher_sha256='abc',image_size=224)
    records=[SimpleNamespace(sample_id='train/1'),SimpleNamespace(sample_id='train/2')]
    payload={'sample_ids':[r.sample_id for r in records],'features':torch.randn(2,192,14,14),
        'manifest':{'checkpoint_sha256':'abc','feature_contract':'aligned_decoder_input_192x14x14_row_major_v1',
                    'augmentation':False,'split':'train_only','image_size':224}}
    torch.save(payload,s.feature_hint_cache)
    return s,records,payload


def test_cache_strict_identity_and_gradients(tmp_path):
    s,records,p=cache(tmp_path)
    state=torch.random.get_rng_state().clone();h=TrainFeatureHints(s,records)
    assert torch.equal(state,torch.random.get_rng_state())
    assert sum(v.numel() for v in h.parameters())==36864
    assert set(h.state_dict())=={'projection.weight'}  # no cache/model in deploy state
    x=p['features'].clone().requires_grad_()
    assert h(x,[r.sample_id for r in records]).item()==pytest.approx(0,abs=1e-6)
    loss=h(x,['train/2','train/1']);loss.backward()
    assert x.grad.abs().sum()>0 and h.projection.weight.grad.abs().sum()>0
    assert not h.values.requires_grad
    with pytest.raises(ValueError):TrainFeatureHints(s,list(reversed(records)))
    p['manifest']['split']='validation';torch.save(p,s.feature_hint_cache)
    with pytest.raises(ValueError):TrainFeatureHints(s,records)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__();self.latent=nn.Parameter(torch.randn(2,192,14,14));self.head=nn.Conv2d(192,1,1)
    def forward(self,*args):return self.head(self.latent),self.latent,self.latent[:,0]
    def router_losses(self):return self.latent.new_zeros(()),self.latent.new_zeros(())
    def operating_loss(self):return self.latent.new_zeros(())


def test_zero_hint_epoch_exactly_preserves_legacy_updates(tmp_path,monkeypatch):
    s,records,p=cache(tmp_path);h=TrainFeatureHints(s,records)
    for k,v in dict(kl_weight=1.,cc_weight=1.5,sim_weight=.25,nss_weight=.1,map_kd_weight=0.,
        phase_dc_weight=0.,router_balance_weight=0.,router_importance_weight=0.,ccd_operating_point_weight=0.,
        feature_hint_current_weight=0.,gradient_clip_norm=1.,log_interval_batches=100).items():setattr(s,k,v)
    monkeypatch.setattr(legacy,'preprocess_vision',lambda *args:{'pixel_values':torch.zeros(1),'image_grid_thw':torch.ones(2,3)})
    monkeypatch.setattr(legacy,'_autocast',lambda *args:nullcontext())
    batch={'sample_ids':[r.sample_id for r in records],'images':[None,None],
           'density':torch.rand(2,1,14,14),'fixation':torch.rand(2,1,14,14)>.9}
    model=ToyStudent();copy=deepcopy(model);loaded=SimpleNamespace(device=torch.device('cpu'),processor=None)
    first=torch.optim.AdamW(model.parameters(),lr=1e-3)
    second=torch.optim.AdamW([{'params':list(copy.parameters())},{'params':list(h.parameters())}],lr=1e-3)
    a=legacy._train_epoch('student',model,[batch],loaded,s,first)
    b=train_hint_epoch(copy,[batch],loaded,s,second,None,h)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,copy.state_dict()[k],rtol=0,atol=0)
    for k in a.keys()-{'epoch_time_sec'}:assert a[k]==pytest.approx(b[k])
    assert all(v.grad is None for v in h.parameters())
    s.feature_hint_current_weight=.1
    train_hint_epoch(copy,[batch],loaded,s,second,None,h)
    assert h.projection.weight.grad.abs().sum()>0
    assert model.state_dict().keys()==copy.state_dict().keys()


def test_hint_profile_does_not_change_inference_architecture():
    root=Path(__file__).resolve().parents[1]/'configs'
    base=load_settings(root/'moe_alpha40_adaptive_keepkd.yaml')
    for name in ['control','cosine','centered']:
        s=load_settings(root/f'moe_alpha40_hint_{name}.yaml')
        assert architecture_label(s)==architecture_label(base)
        assert not s.augmentation_enabled and s.top_k==2 and s.router_backend=='optical'
        assert s.fusion_alpha_min==.4 and s.active_size==478 and s.expert_size==224
        assert s.language_optical_phase_zero_order_intensity_min==.2
        assert s.language_optical_phase_zero_order_intensity_max==.3
        assert s.initialization_checkpoint_sha256==base.initialization_checkpoint_sha256


def test_centered_hint_is_sensitive_to_local_structure_not_common_bias():
    torch.manual_seed(123)
    x=torch.randn(2,192,14,14);y=torch.randn_like(x)
    bias=torch.randn(2,192,1,1)*100
    raw=feature_hint_loss(x+bias,y+bias,'cosine')
    centered=feature_hint_loss(x+bias,y+bias,'spatial_centered_cosine')
    assert raw < .001 and centered > .9
    torch.testing.assert_close(centered,feature_hint_loss(x,y,'spatial_centered_cosine'),rtol=1e-5,atol=1e-5)
    x.requires_grad_();feature_hint_loss(x,y,'spatial_centered_cosine').backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert feature_hint_loss(torch.zeros_like(x),torch.zeros_like(y),'spatial_centered_cosine').isfinite()
    with pytest.raises(ValueError):feature_hint_loss(x,y,'unknown')
