from pathlib import Path
import random
import numpy as np
import pytest
import torch
from LightGenV2.tasks.t03_saliency.sam_training import sam_step,train_sam_epoch
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


def test_sam_perturbs_only_electronic_but_updates_optics_and_steps_once():
    e=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64))
    p=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
    opt=torch.optim.SGD([{'params':[e],'name':'electronic'},
                         {'params':[p],'name':'feature_phase'}],lr=.1)
    seen=[];steps=[]
    opt.register_step_post_hook(lambda *args:steps.append((e.item(),p.item())))
    def closure():
        seen.append((e.item(),p.item()))
        loss=(e+2*p).square()/2
        return loss,{'loss':loss}
    values,increase=sam_step(opt,closure,.05,torch.device('cpu'))
    assert seen[0]==(1.,2.) and seen[1][1]==2.
    assert seen[1][0]==pytest.approx(1.05)
    assert e.item()==pytest.approx(.495) and p.item()==pytest.approx(.99)
    assert len(steps)==1 and values['loss'].item()==12.5
    assert increase.item()==pytest.approx((5.05**2-5**2)/2)


def test_same_randomness_and_exact_restore_after_second_failure():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    e=torch.nn.Parameter(torch.tensor([.1,.3]))
    old=e.detach().clone();opt=torch.optim.SGD([{'params':[e],'name':'electronic'}],lr=.1)
    draws=[];after=[]
    def closure():
        noise=torch.rand(2);a=random.random();b=np.random.random()
        draws.append((noise,a,b));after.append(torch.get_rng_state().clone())
        if len(draws)==2:raise RuntimeError('deliberate second pass failure')
        loss=(e*noise).square().sum()+a+b
        return loss,{'loss':loss}
    with pytest.raises(RuntimeError,match='deliberate'):sam_step(opt,closure,.05,torch.device('cpu'))
    torch.testing.assert_close(e,old,rtol=0,atol=0)
    torch.testing.assert_close(draws[0][0],draws[1][0],rtol=0,atol=0)
    assert draws[0][1:]==draws[1][1:]
    assert torch.equal(torch.get_rng_state(),after[0])
    assert not opt.state


def test_zero_sam_is_one_normal_update_and_legacy_epoch(monkeypatch):
    from types import SimpleNamespace
    e=torch.nn.Parameter(torch.tensor([.2,.4]))
    opt=torch.optim.SGD([{'params':[e],'name':'electronic'}],lr=.1)
    def closure():
        loss=e.square().sum()
        return loss,{'loss':loss}
    sam_step(opt,closure,0,torch.device('cpu'))
    torch.testing.assert_close(e,torch.tensor([.16,.32]))
    sentinel={'ok':True}
    import LightGenV2.tasks.t03_saliency.sam_training as module
    monkeypatch.setattr(module.legacy,'_train_epoch',lambda *args,**kwargs:sentinel)
    assert train_sam_epoch(None,None,None,SimpleNamespace(sam_rho=0),None) is sentinel


@pytest.mark.parametrize('name,rho', [('sam001', .01), ('sam005', .05), ('sam010', .1)])
def test_sam_configuration_only_changes_training_and_preserves_best_source(name,rho):
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_sam_control.yaml')
    b=load_settings(root/f'moe_alpha40_{name}.yaml')
    reference=load_settings(root/'moe_alpha40_viewreg_cffn_kd2.yaml')
    assert a.sam_rho==0 and b.sam_rho==rho
    assert b.output_dir.name==f'moe_alpha40_{name}_seed42'
    assert architecture_label(a)==architecture_label(b)==architecture_label(reference)
    for k in ['initialization_checkpoint_sha256','student_epochs','student_learning_rate',
              'phase_learning_rate','ema_decay','weight_decay','distillation_initial_weight',
              'distillation_final_weight','fusion_alpha_min','top_k','router_backend',
              'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(a,k)==getattr(b,k)
    assert a.initialization_checkpoint_sha256=='c88e1a41febc878cf87d6c80d4e27d7ac0c6bddc11d73a29497490d2e841ad73'
    assert a.student_epochs==50 and not a.augmentation_enabled


def test_early_sam_only_changes_radius_and_output():
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_viewreg_cffn_kd2.yaml')
    b=load_settings(root/'moe_alpha40_viewreg_sam005.yaml')
    assert a.sam_rho==0 and b.sam_rho==.05
    assert architecture_label(a)==architecture_label(b)
    for key in ['initialization_checkpoint_sha256','student_epochs','student_learning_rate',
                'phase_learning_rate','router_learning_rate','ema_decay','weight_decay',
                'augmentation_enabled','augmentation_end_epoch','augmentation_apply_probability',
                'distillation_initial_weight','distillation_final_weight','distillation_end_epoch',
                'fusion_alpha_min','top_k','router_backend','initialize_ffn_on_warmstart']:
        assert getattr(a,key)==getattr(b,key)
    assert b.student_epochs==80 and b.augmentation_enabled
    assert b.initialization_checkpoint_sha256=='de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea'
    assert b.output_dir.name=='moe_alpha40_viewreg_sam005_seed42'
