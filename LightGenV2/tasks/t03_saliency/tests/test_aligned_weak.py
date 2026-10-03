from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from LightGenV2.tasks.t03_saliency.training_support import AlignedWeakLoader, TrainTeacherMaps, warp_density
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


def fixture(tmp_path):
    values = torch.arange(1,65).float().reshape(1,1,8,8)
    values = torch.cat([values, values.flip(-1)])
    density = values / values.sum((-2,-1),keepdim=True)
    s = SimpleNamespace(random_seed=42, augmentation_enabled=True, augmentation_mode='aligned_weak',
        distillation_cache=tmp_path/'teacher.pt', distillation_teacher_sha256='abc', image_size=8,
        crop_scale_min=.8, horizontal_flip_probability=.5, brightness_jitter=0., contrast_jitter=0.)
    ids = ['train/1','train/2']
    torch.save({'sample_ids':ids,'logits':density.log(),
        'manifest':{'checkpoint_sha256':'abc','image_size':8,'augmentation':False}},s.distillation_cache)
    teacher = TrainTeacherMaps(s,[SimpleNamespace(sample_id=i) for i in ids])
    batch = {'sample_ids':ids,'images':[Image.fromarray(np.repeat((v[0].numpy()*3).astype('uint8')[...,None],3,axis=-1)) for v in values],
        'density':density,'fixation':(values % 5 == 0).float()}
    return s,teacher,batch


def test_transform_alignment_mass_rng_and_no_mutation(tmp_path):
    s,t,b = fixture(tmp_path)
    before=b['density'].clone(); pixels=np.array(b['images'][0]).copy()
    state=torch.random.get_rng_state().clone()
    loader=AlignedWeakLoader([b],s,t)
    out=next(iter(loader))
    probs=t.get(b['sample_ids'],'cpu').flatten(1).softmax(-1).reshape_as(before)
    torch.testing.assert_close(probs,out['density'],rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(out['density'].sum((-2,-1)),torch.ones(2,1))
    assert set(out['fixation'].unique().tolist()) <= {0.,1.}
    assert torch.equal(state,torch.random.get_rng_state())
    assert torch.equal(before,b['density']) and np.array_equal(pixels,np.array(b['images'][0]))
    repeat=next(iter(AlignedWeakLoader([b],s,t)))
    torch.testing.assert_close(out['density'],repeat['density'],rtol=0,atol=0)
    with pytest.raises(ValueError): t.get(list(reversed(b['sample_ids'])),'cpu')
    loader.enabled=False
    plain=next(iter(loader))
    assert plain['images'] is b['images']
    torch.testing.assert_close(t.get(b['sample_ids'],'cpu'),t.get_raw(b['sample_ids']),rtol=0,atol=0)


def test_only_train_and_aligned_geometry(tmp_path):
    s,t,b = fixture(tmp_path)
    with pytest.raises(ValueError): t.get(b['sample_ids'],'cpu')
    b['sample_ids'][0]='validation/1'
    with pytest.raises(ValueError): next(iter(AlignedWeakLoader([b],s,t)))
    b['sample_ids'][0]='train/1'
    b['images'][0]=b['images'][0].resize((7,8))
    with pytest.raises(ValueError): next(iter(AlignedWeakLoader([b],s,t)))


def test_warp_is_probability_not_logit_interpolation():
    d=torch.tensor([[[[.1,.2],[.3,.4]]]])
    torch.testing.assert_close(warp_density(d,(0,0,2,2),True),d.flip(-1))
    cropped=warp_density(d,(1,0,2,2),False)
    expected=torch.tensor([[[[1/6,1/6],[1/3,1/3]]]])
    torch.testing.assert_close(cropped,expected)
    with pytest.raises(ValueError): warp_density(torch.zeros_like(d),(0,0,2,2),False)


def test_profiles_keep_inference_contract():
    root=Path(__file__).resolve().parents[1]/'configs'
    old=load_settings(root/'moe_alpha40_cffn_control.yaml')
    for name in ['control','cffn','cffn_kd2']:
        s=load_settings(root/f'moe_alpha40_viewreg_{name}.yaml')
        assert s.augmentation_mode=='aligned_weak' and s.augmentation_end_epoch==60
        assert s.student_epochs==80 and s.ema_decay==.995
        assert s.fusion_alpha_min==.4 and not s.reset_fusion_on_warmstart
        assert s.router_backend=='optical' and s.top_k==2
        assert s.language_optical_phase_zero_order_intensity_min==.2
        assert s.language_optical_phase_zero_order_intensity_max==.3
        assert s.expert_size==old.expert_size and s.pixel_pitch_um==old.pixel_pitch_um
        assert s.initialization_checkpoint_sha256=='de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea'
        assert architecture_label(s)==architecture_label(load_settings(root/('moe_alpha40_cffn_control.yaml' if name=='control' else 'moe_alpha40_cffn_d1.yaml')))


def test_partial_augmentation_identity_default_and_alignment(tmp_path):
    s,t,b = fixture(tmp_path)
    default = AlignedWeakLoader([b],s,t)
    original_out = next(iter(default))
    s.augmentation_apply_probability = 1.0
    explicit = AlignedWeakLoader([b],s,t)
    explicit_out = next(iter(explicit))
    assert default.rng.getstate() == explicit.rng.getstate()
    torch.testing.assert_close(original_out['density'],explicit_out['density'],rtol=0,atol=0)
    assert all(np.array_equal(np.array(a),np.array(c)) for a,c in zip(original_out['images'],explicit_out['images']))
    s.augmentation_apply_probability = 0.0
    identity = AlignedWeakLoader([b],s,t)
    out = next(iter(identity))
    assert all(a is c for a,c in zip(out['images'],b['images']))
    torch.testing.assert_close(out['density'],b['density'],rtol=0,atol=0)
    torch.testing.assert_close(out['fixation'],b['fixation'],rtol=0,atol=0)
    torch.testing.assert_close(t.get(b['sample_ids'],'cpu'),t.get_raw(b['sample_ids']),rtol=0,atol=0)
    assert identity.epoch_images == 2 and identity.epoch_augmented_images == 0
    s.augmentation_apply_probability = .5
    mixed = AlignedWeakLoader([b]*50,s,t)
    unchanged = 0
    for out in mixed:
        probs=t.get(b['sample_ids'],'cpu').flatten(1).softmax(-1).reshape_as(b['density'])
        torch.testing.assert_close(probs,out['density'],rtol=1e-5,atol=1e-7)
        unchanged += sum(a is c for a,c in zip(out['images'],b['images']))
    assert mixed.epoch_images == 100
    assert 20 < mixed.epoch_augmented_images < 80
    assert unchanged == 100-mixed.epoch_augmented_images
    mixed.enabled=False
    list(mixed)
    assert mixed.epoch_images == 100 and mixed.epoch_augmented_images == 0


def test_mix50_profile_is_training_only_single_change():
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_viewreg_cffn_kd2.yaml')
    b=load_settings(root/'moe_alpha40_viewreg_mix50.yaml')
    assert a.augmentation_apply_probability == 1 and b.augmentation_apply_probability == .5
    assert architecture_label(a) == architecture_label(b)
    for name in ['initialization_checkpoint_sha256','student_epochs','staged_polish_start',
                 'augmentation_end_epoch','fusion_alpha_min','distillation_initial_weight',
                 'distillation_final_weight','language_optical_phase_zero_order_intensity_min',
                 'language_optical_phase_zero_order_intensity_max','top_k','router_backend',
                 'expert_size','pixel_pitch_um','ema_decay']:
        assert getattr(a,name) == getattr(b,name)
