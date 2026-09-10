from contextlib import nullcontext
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from LightGenV2.tasks.t03_saliency.relational_distillation import pairwise_loss, RelationalTargets, configure
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.sam_training import train_sam_epoch
from LightGenV2.tasks.t03_saliency.training_support import ModelEMA
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy


def test_basis_invariance_spatial_sensitivity_and_detached_teacher():
    torch.manual_seed(10)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    target = torch.einsum('ij,bjhw->bihw', q, x.detach()) * 3 + torch.randn(2,8,1,1)
    target.requires_grad_()
    assert pairwise_loss(x, target) < 1e-12
    loss = pairwise_loss(x, target.flip(-1))
    assert loss > .01
    loss.backward()
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    assert target.grad is None
    assert pairwise_loss(torch.zeros_like(x), torch.zeros_like(x)) == 0
    with pytest.raises(ValueError): pairwise_loss(x, target[:1])


def cache_setup(tmp_path):
    ids = ['train/a','train/b']
    values = torch.randn(2,192,14,14)
    manifest = dict(checkpoint_sha256='1'*64, feature_contract='aligned_decoder_input_192x14x14_row_major_v1',
                    split='train_only', augmentation=False, image_size=224)
    path = tmp_path/'cache.pt'
    torch.save(dict(sample_ids=ids, features=values, manifest=manifest), path)
    options = dict(cache_file=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   initial_weight=1.,final_weight=.1,end_epoch=30)
    s = SimpleNamespace(relational_distillation=options, distillation_teacher_sha256='1'*64,image_size=224)
    return s,[SimpleNamespace(sample_id=k) for k in ids],values


def test_cache_identity_sha_and_no_parameters(tmp_path):
    s,records,values = cache_setup(tmp_path)
    before = torch.get_rng_state().clone()
    cache = RelationalTargets(s,records)
    assert torch.equal(before,torch.get_rng_state())
    assert not isinstance(cache,nn.Module)
    assert cache.provenance['inference_parameters_added']==0
    assert cache.loss(values,[r.sample_id for r in records]) < 1e-12
    with pytest.raises(KeyError): cache.loss(values,['test/a','train/b'])
    with pytest.raises(ValueError): RelationalTargets(s,list(reversed(records)))
    s.relational_distillation['cache_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA256'): RelationalTargets(s,records)


def test_profile_preserves_inference_and_rejects_confounded_modes():
    root = Path(__file__).resolve().parents[1]/'configs'
    old = load_settings(root/'moe_alpha40_extra_control.yaml')
    new = load_settings(root/'moe_alpha40_relational_kd.yaml')
    assert architecture_label(old)==architecture_label(new)
    a,b=vars(old).copy(),vars(new).copy()
    for key in ('config_path','config_file','output_dir','relational_distillation'):
        a.pop(key,None);b.pop(key,None)
    # Paths referring to the config may be stored by shared settings; the effective
    # training and model contract below must remain strictly equal.
    for key in ('initialization_checkpoint_sha256','student_epochs','sam_rho','ema_decay','top_k',
                'router_backend','fusion_alpha_min','phase_learning_rate','student_learning_rate',
                'distillation_initial_weight','distillation_final_weight','ccd_normalization',
                'electronic_width','electronic_ffn_hidden_width','reset_fusion_on_warmstart'):
        assert a[key]==b[key]
    assert not old.relational_distillation and new.student_epochs==40
    assert new.initialization_checkpoint_sha256.startswith('87ad4db5')
    for field,value in [('augmentation_enabled',True),('sam_rho',0),('reset_fusion_on_warmstart',True),
                        ('feature_pretraining',{'enabled':True})]:
        bad=deepcopy(new);setattr(bad,field,value)
        with pytest.raises(ValueError): configure(bad,new.relational_distillation,root)
    for field,value in [('initial_weight',float('nan')),('final_weight',2),('end_epoch',True)]:
        options=dict(new.relational_distillation);options[field]=value
        with pytest.raises(ValueError): configure(deepcopy(new),options,root)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=nn.Conv2d(3,192,1)
        self.head=nn.Conv2d(192,1,1)
    def forward(self,x,*args):
        spatial=self.core(x)
        return self.head(spatial),spatial,None
    def router_losses(self):
        z=self.core.weight.new_zeros(())
        return z,z


def test_sam_relations_execute_twice_update_once_and_keep_state_keys(tmp_path,monkeypatch):
    s,records,_=cache_setup(tmp_path)
    for k,v in dict(sam_rho=.05,gradient_clip_norm=1.,relational_current_weight=1.,
        kl_weight=1.,cc_weight=.5,sim_weight=.25,nss_weight=.1,map_kd_weight=0.,distillation_loss='spatial_cc',
        router_balance_weight=0.,router_importance_weight=0.,phase_dc_weight=0.,log_interval_batches=100).items():
        setattr(s,k,v)
    targets=RelationalTargets(s,records)
    model=Toy();keys=set(model.state_dict());before=deepcopy(model.state_dict())
    opt=torch.optim.AdamW([{'params':model.core.parameters(),'name':'electronic'},
                           {'params':model.head.parameters(),'name':'saliency_head'}],lr=.001)
    ema=ModelEMA(model,.9);count=[]
    handle=opt.register_step_post_hook(lambda *a:(count.append(1),ema.update(*a)))
    real=targets.loss;calls=[]
    def loss(*args):calls.append(1);return real(*args)
    targets.loss=loss
    monkeypatch.setattr(legacy,'_autocast',lambda *a:nullcontext())
    monkeypatch.setattr(legacy,'preprocess_vision',lambda p,images,d:dict(pixel_values=images,image_grid_thw=None))
    batch=dict(images=torch.randn(2,3,14,14),sample_ids=[r.sample_id for r in records],
               density=torch.rand(2,1,14,14),fixation=torch.rand(2,1,14,14)>.9)
    result=train_sam_epoch(model,[batch],SimpleNamespace(device=torch.device('cpu'),processor=None),
                           s,opt,relation_targets=targets)
    handle.remove()
    assert len(calls)==2 and len(count)==1 and result['samples']==2 and result['relational_loss']>0
    assert set(model.state_dict())==keys
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
