from copy import deepcopy
from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from LightGenV2.tasks.t03_saliency.feature_pretraining import (
    configure, FixedFeatureTargets, initialize_teacher_decoder, prepare_epoch, train_epoch,
    configure_router_path_optimizer)
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.training_support import ModelEMA
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy


def setup(tmp_path):
    records = [SimpleNamespace(sample_id=f'train/{i}') for i in range(2)]
    teacher = nn.Conv2d(192,1,1)
    head = tmp_path/'head.pt'
    torch.save({'head':{'decoder.'+k:v for k,v in teacher.state_dict().items()}},head)
    digest = hashlib.sha256(head.read_bytes()).hexdigest()
    features = torch.randn(2,192,14,14)
    cache = tmp_path/'features.pt'
    payload = {'sample_ids':[r.sample_id for r in records], 'features':features,
               'manifest':{'checkpoint_sha256':digest,'feature_contract':'aligned_decoder_input_192x14x14_row_major_v1',
                           'augmentation':False,'split':'train_only','image_size':224}}
    torch.save(payload,cache)
    options = dict(enabled=True,cache_file=str(cache),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                   head_checkpoint=str(head),head_checkpoint_sha256=digest,frozen_head_epochs=2,fade_end_epoch=5,
                   initial_weight=2.,joint_weight=.2,mean_weight=.05,joint_lr_multiplier=.2)
    settings = SimpleNamespace(feature_pretraining=options, image_size=224,student_epochs=8,sam_rho=.05)
    return settings,records,payload,teacher


def test_train_identity_sha_and_loss_statistics(tmp_path):
    settings,records,payload,_ = setup(tmp_path)
    state = torch.get_rng_state().clone()
    targets = FixedFeatureTargets(settings,records)
    assert torch.equal(state,torch.get_rng_state())
    ids = payload['sample_ids']
    x = payload['features'].clone().requires_grad_()
    total,local,mean = targets.loss(x,ids,.05)
    assert total.item() < 1e-12
    total,local,mean = targets.loss(x+3,ids,.05)
    assert local.item() < 1e-10 and mean.item() > 0
    total.backward()
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    with pytest.raises(KeyError): targets.loss(x,['test/0','train/1'],.05)
    with pytest.raises(ValueError): FixedFeatureTargets(settings,list(reversed(records)))
    settings.feature_pretraining['cache_sha256'] = '0'*64
    with pytest.raises(ValueError,match='SHA256'): FixedFeatureTargets(settings,records)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(1,1)
        self.latent = nn.Parameter(torch.randn(2,192,14,14))
        self.head = nn.Conv2d(192,1,1)
    def forward(self,*args):
        spatial = self.latent*self.core.weight.reshape(1,1,1,1)+self.core.bias.reshape(1,1,1,1)
        return self.head(spatial),spatial,None
    def router_losses(self):return self.latent.new_zeros(()),self.latent.new_zeros(())


def test_head_transfer_freeze_joint_sam_ema_and_retention(tmp_path,monkeypatch):
    s,records,payload,teacher = setup(tmp_path)
    model = Toy()
    old_core = deepcopy(model.core.state_dict())
    keys = set(model.state_dict())
    report = initialize_teacher_decoder(model,s)
    assert report['inference_parameters_added'] == 0 and set(model.state_dict())==keys
    for key,value in old_core.items(): assert torch.equal(value,model.core.state_dict()[key])
    for key,value in teacher.state_dict().items():assert torch.equal(value,model.head.state_dict()[key])
    optim = torch.optim.AdamW([{'params':list(model.core.parameters())+[model.latent],'name':'electronic','lr':.001},
                              {'params':list(model.head.parameters()),'name':'saliency_head','lr':.001}])
    ema = ModelEMA(model,.9)
    hook = optim.register_step_post_hook(ema.update)
    targets = FixedFeatureTargets(s,records)
    for key,value in dict(kl_weight=1.,cc_weight=.5,sim_weight=.25,nss_weight=.1,map_kd_weight=0.,
        distillation_loss='kl',phase_dc_weight=0.,router_balance_weight=0.,router_importance_weight=0.,
        ccd_operating_point_weight=0.,gradient_clip_norm=1.,log_interval_batches=100).items():setattr(s,key,value)
    monkeypatch.setattr(legacy,'preprocess_vision',lambda *args:{'pixel_values':torch.zeros(1),'image_grid_thw':torch.ones(2,3)})
    monkeypatch.setattr(legacy,'_autocast',lambda *args:nullcontext())
    batch={'sample_ids':payload['sample_ids'],'images':[None,None],'density':torch.rand(2,1,14,14),
           'fixation':torch.rand(2,1,14,14)>.9}
    loaded=SimpleNamespace(device=torch.device('cpu'),processor=None)
    current,report=prepare_epoch(model,optim,s,1)
    assert current.sam_rho==0 and report['head_frozen'] and not any(p.requires_grad for p in model.head.parameters())
    assert s.sam_rho==.05
    before=deepcopy(model.head.state_dict())
    train_epoch(model,[batch],loaded,current,optim,None,targets,report['feature_weight'])
    assert not torch.equal(old_core['weight'],model.core.weight)
    for k,v in before.items():assert torch.equal(v,model.head.state_dict()[k]) and torch.equal(v,ema.shadow['head'][k])
    current,report=prepare_epoch(model,optim,s,3)
    assert current.sam_rho==.05 and all(p.requires_grad for p in model.head.parameters())
    assert optim.param_groups[0]['lr']==pytest.approx(.0002)
    train_epoch(model,[batch],loaded,current,optim,None,targets,report['feature_weight'])
    assert not torch.equal(before['weight'],model.head.weight)
    assert model.head.weight in optim.state
    _,report=prepare_epoch(model,optim,s,5)
    assert report['feature_weight']==0
    _,report=prepare_epoch(model,optim,s,8)
    assert optim.param_groups[0]['lr']==pytest.approx(.00001)
    assert set(model.state_dict())==keys
    hook.remove()


def test_profile_architecture_and_invalid_options():
    root=Path(__file__).resolve().parents[1]/'configs'
    s=load_settings(root/'moe_alpha40_feature_pretrain.yaml')
    base=load_settings(root/'moe_alpha40_extra_control.yaml')
    assert architecture_label(s)==architecture_label(base)
    assert s.fusion_alpha_min==.4 and s.active_size==478 and s.expert_size==224
    assert s.router_backend=='optical' and s.top_k==2
    assert s.language_optical_phase_zero_order_intensity_min==.2 and s.language_optical_phase_zero_order_intensity_max==.3
    assert s.initialization_checkpoint_sha256==base.initialization_checkpoint_sha256
    options=deepcopy(s.feature_pretraining)
    s.augmentation_enabled=True
    with pytest.raises(ValueError):configure(s,options,root)
    s.augmentation_enabled=False
    options['head_checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):configure(s,options,root)
    options=deepcopy(s.feature_pretraining);options['frozen_head_epochs']=True
    with pytest.raises(ValueError):configure(s,options,root)


def test_strong_pretraining_is_only_an_initial_feature_weight_change():
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_feature_pretrain.yaml')
    b=load_settings(root/'moe_alpha40_feature_pretrain_strong.yaml')
    allowed={'config','config_path','output_dir','feature_pretraining'}
    assert {key for key in vars(a) if getattr(a,key)!=getattr(b,key)} <= allowed
    assert {key for key in a.feature_pretraining if a.feature_pretraining[key]!=b.feature_pretraining[key]}=={'initial_weight'}
    assert a.feature_pretraining['initial_weight']==2. and b.feature_pretraining['initial_weight']==10.
    assert architecture_label(a)==architecture_label(b)


def test_stable_router_freezes_both_input_adapters_and_phase_then_unfreezes():
    s=load_settings(Path(__file__).resolve().parents[1]/'configs/moe_alpha40_feature_pretrain_stable_router.yaml')
    model=nn.Module();model.core=nn.Module();model.core.hybrid=nn.Module();h=model.core.hybrid
    h.input_adapter=nn.Linear(4,4);h.input_norm=nn.LayerNorm(4)
    h.optical_branch=nn.Module();h.optical_branch.core=nn.Module();o=h.optical_branch.core
    o.input_adapter=nn.Linear(4,4);o.input_norm=nn.LayerNorm(4);o.router=nn.Linear(4,4)
    model.core.other=nn.Parameter(torch.ones(3));model.head=nn.Linear(4,1)
    frozen_inputs=[p for m in [h.input_adapter,h.input_norm,o.input_adapter,o.input_norm] for p in m.parameters()]
    optim=torch.optim.AdamW([
        {'name':'electronic','params':[model.core.other]+frozen_inputs,'lr':.0003},
        {'name':'optical_router','params':list(o.router.parameters()),'lr':.00005},
        {'name':'saliency_head','params':list(model.head.parameters()),'lr':.0003}])
    keys=set(model.state_dict());before=deepcopy(model.state_dict())
    report=configure_router_path_optimizer(model,optim,s)
    assert report['router_input_parameter_count']==56
    assert report['router_input_sam_perturbation'] is False
    assert len(report['router_input_parameter_names'])==8
    from LightGenV2.tasks.t03_saliency.sam_training import ELECTRONIC_GROUPS
    assert 'router_input' not in ELECTRONIC_GROUPS
    x=torch.randn(8,4)
    def routing():return o.router(o.input_norm(o.input_adapter(h.input_norm(h.input_adapter(x)))))
    source=routing().detach().clone()
    current,stage=prepare_epoch(model,optim,s,1)
    assert stage['router_path_frozen']
    for group in optim.param_groups:
        for p in group['params']:
            if p.requires_grad:p.grad=torch.ones_like(p)
    optim.step();optim.zero_grad(set_to_none=True)
    assert not torch.equal(model.core.other,before['core.other'])
    torch.testing.assert_close(routing(),source,rtol=0,atol=0)
    for k,v in model.state_dict().items():
        if k!='core.other':assert torch.equal(v,before[k])
    _,stage=prepare_epoch(model,optim,s,16)
    assert not stage['router_path_frozen']
    group=next(g for g in optim.param_groups if g['name']=='router_input')
    assert group['lr']==pytest.approx(1e-6) and all(p.requires_grad for p in frozen_inputs)
    for group in optim.param_groups:
        for p in group['params']:p.grad=torch.ones_like(p)
    optim.step()
    assert all(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert set(model.state_dict())==keys


def test_stable_router_preserves_inference_and_warmstart_source():
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_feature_pretrain.yaml')
    b=load_settings(root/'moe_alpha40_feature_pretrain_stable_router.yaml')
    assert architecture_label(a)==architecture_label(b)
    assert a.initialization_checkpoint_sha256==b.initialization_checkpoint_sha256
    assert {key for key in vars(a) if getattr(a,key)!=getattr(b,key)} <= {'config','config_path','output_dir','feature_pretraining'}
    assert b.feature_pretraining['freeze_router_path']
    options=deepcopy(b.feature_pretraining);options['router_input_learning_rate']=.001
    with pytest.raises(ValueError):configure(b,options,root)


def test_joint_only_profile_starts_unfrozen_without_repeating_pretraining():
    root=Path(__file__).resolve().parents[1]/'configs'
    s=load_settings(root/'moe_alpha40_feature_joint_router_low_lr.yaml')
    assert s.student_epochs==45 and s.feature_pretraining['frozen_head_epochs']==0
    assert s.initialization_checkpoint_sha256=='b2d8e0d9d903efa1de66664e02af07fd9ff9310e53c1f4db71889b0b99508fba'
    model=Toy();optim=torch.optim.AdamW([
        {'name':'electronic','params':list(model.core.parameters())+[model.latent],'lr':.0003},
        {'name':'saliency_head','params':list(model.head.parameters()),'lr':.0003}])
    current,stage=prepare_epoch(model,optim,s,1)
    assert not stage['head_frozen'] and not stage['router_path_frozen']
    assert current.sam_rho==.05 and stage['feature_weight']==.2
    assert stage['lr_electronic']==pytest.approx(.00006)
    _,stage=prepare_epoch(model,optim,s,25)
    assert stage['feature_weight']==0
