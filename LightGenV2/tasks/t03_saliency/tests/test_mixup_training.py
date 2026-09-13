from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t03_saliency.mixup_training import prepare, paired_loss, validate
from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


def inputs():
    return {'pixel_values':torch.arange(24,dtype=torch.float32).reshape(8,3),
            'image_grid_thw':torch.tensor([[1,2,2],[1,2,2]])}


def selected_seed(x, options):
    for seed in range(100):
        torch.manual_seed(seed)
        out,mix=prepare(x,['train/a','train/b'],options,enabled=True)
        if mix is not None:return seed,out,mix
    raise AssertionError('No selected batch')


def test_mixup_is_exact_affine_input_combination_and_preserves_packing():
    opts=validate(dict(alpha=.2,probability=.5,end_epoch=15));x=inputs()
    old=x['pixel_values'].clone();seed,out,(lam,idx)=selected_seed(x,opts)
    assert idx.tolist()==[1,0] and .5 <= lam <= 1
    rows=old.reshape(2,4,3)
    torch.testing.assert_close(out['pixel_values'],(lam*rows+(1-lam)*rows[idx]).reshape_as(old))
    assert torch.equal(x['pixel_values'],old) and out['image_grid_thw'] is x['image_grid_thw']
    torch.manual_seed(seed);repeat,mix=prepare(x,['train/a','train/b'],opts,enabled=True)
    assert torch.equal(out['pixel_values'],repeat['pixel_values']) and mix[0]==lam


def test_disabled_no_rng_and_eval_identity_rejected():
    x=inputs();opts=dict(alpha=.2,probability=.5,end_epoch=15)
    state=torch.get_rng_state().clone()
    out,mix=prepare(x,['val/a','val/b'],opts,enabled=False)
    assert out is x and mix is None and torch.equal(state,torch.get_rng_state())
    with pytest.raises(ValueError,match='train identities'):
        prepare(x,['train/a','val/b'],opts,enabled=True)
    seed,_,_=selected_seed(x,opts)
    bad=inputs();bad['image_grid_thw'][1]=torch.tensor([1,1,4])
    torch.manual_seed(seed)
    with pytest.raises(ValueError,match='identically packed'):prepare(bad,['train/a','train/b'],opts,enabled=True)


def test_endpoint_losses_pair_fixations_teacher_and_backpropagate():
    x=torch.tensor([[.2,.8],[.7,.3]],requires_grad=True)
    y=torch.tensor([[.1,.9],[.4,.6]]);f=torch.tensor([[1.,0],[0,1.]])
    t=torch.tensor([[2.,3.],[4.,5.]])
    seen=[]
    def loss(pred,density,fixation,settings,teacher_logits):
        seen.append((density.clone(),fixation.clone(),teacher_logits.clone()))
        value=(pred-density).square().mean()+(pred*fixation).mean()+(pred-teacher_logits).square().mean()
        return value,{'loss':value,'cc':value*2}
    lam=.7;idx=torch.tensor([1,0])
    actual,pieces=paired_loss(loss,x,y,f,None,t,(lam,idx))
    assert torch.equal(seen[0][1],f) and torch.equal(seen[1][1],f[idx])
    assert torch.equal(seen[1][2],t[idx])
    expected=lam*loss(x,y,f,None,t)[0]+(1-lam)*loss(x,y[idx],f[idx],None,t[idx])[0]
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(pieces['loss'],actual)
    actual.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_zero_mix_keeps_single_loss():
    calls=[];x=torch.ones(1,requires_grad=True)
    def loss(*args,**kwargs):calls.append(1);return x,{'loss':x}
    out,pieces=paired_loss(loss,x,None,None,None,None,None)
    assert out is x and len(calls)==1


@pytest.mark.parametrize('key,value',[('alpha',0),('alpha',.5),('alpha',float('nan')),
    ('probability',.6),('probability',float('inf')),('end_epoch',True),('end_epoch',1.5)])
def test_invalid_options(key,value):
    options=dict(alpha=.2,probability=.25,end_epoch=15);options[key]=value
    with pytest.raises(ValueError):validate(options)


def test_profile_preserves_architecture_noise_and_serializes(tmp_path):
    import yaml
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_sam_batch8_crosssample_20260913.yaml')
    b=load_settings(root/'moe_alpha40_mixup_20260913.yaml')
    assert not a.mixup and not b.mixup_active
    assert b.mixup==dict(alpha=.2,probability=.25,end_epoch=15)
    assert b.student_epochs==20 and b.staged_polish_start==16
    assert architecture_label(a)==architecture_label(b)
    for k in ['student_batch_size','student_learning_rate','phase_learning_rate','router_learning_rate',
              'ema_decay','sam_rho','fusion_alpha_min','router_backend','top_k','active_size','expert_size',
              'pixel_pitch_um','distillation_teacher_sha256','router_balance_estimator']:
        assert getattr(a,k)==getattr(b,k),k
    for k in vars(a):
        if k.startswith('language_optical_') or k.startswith('optical_router_'):
            assert getattr(a,k)==getattr(b,k),k
    b.output_dir=tmp_path;save_resolved_config(b)
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['training']['mixup']==b.mixup
    path=tmp_path/'invalid.yaml'
    path.write_text(f'base_config: {(root/"moe_alpha40_mixup_20260913.yaml").as_posix()}\ntraining:\n  noise_consistency_weight: 1\n')
    with pytest.raises(ValueError,match='MixUp'):load_settings(path)
