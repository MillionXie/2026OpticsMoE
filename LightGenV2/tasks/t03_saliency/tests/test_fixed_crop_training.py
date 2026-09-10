from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from LightGenV2.tasks.t03_saliency.tests.test_fixed_crop_teacher import fixture
from LightGenV2.tasks.t03_saliency.fixed_crop_teacher import BOX, crop_image
from LightGenV2.tasks.t03_saliency.fixed_crop_training import FixedCropLoader, FixedCropTargets, configure
from LightGenV2.tasks.t03_saliency.training_support import TrainTeacherMaps, warp_density
from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


ROOT=Path(__file__).resolve().parents[1]/'configs'


def setup(tmp_path, mode='actual', probability=1.):
    images,records,payload=fixture()
    s=load_settings(ROOT/'moe_alpha40_fixed_crop_actual.yaml')
    s.fixed_crop_distillation=dict(s.fixed_crop_distillation, mode=mode, apply_probability=probability)
    s.distillation_cache=tmp_path/'original.pt'
    raw=torch.randn(2,1,224,224)
    torch.save(dict(logits=raw,sample_ids=payload['sample_ids'],manifest=dict(
        image_size=224,augmentation=False,checkpoint_sha256=s.distillation_teacher_sha256)),s.distillation_cache)
    teacher=TrainTeacherMaps(s,records)
    target=SimpleNamespace(payload=payload,index=teacher.index)
    density=torch.rand(2,1,224,224);density/=density.sum((-2,-1),keepdim=True)
    fixation=torch.zeros_like(density);fixation[:,:,50,60]=1
    batch=dict(sample_ids=payload['sample_ids'],images=images,density=density,fixation=fixation)
    return s,teacher,target,batch


def test_actual_and_proxy_change_only_teacher_not_inputs_gt_or_rng(tmp_path):
    s,t,c,b=setup(tmp_path);original=deepcopy(b)
    rng=torch.random.get_rng_state().clone()
    loader=FixedCropLoader([b],s,t,c);it=iter(loader);actual=next(it)
    for im,source in zip(actual['images'],b['images']):
        assert im.tobytes()==crop_image(source).tobytes()
    torch.testing.assert_close(actual['density'],warp_density(b['density'],BOX,False))
    torch.testing.assert_close(t.get(b['sample_ids'],'cpu'),c.payload['logits'].float(),rtol=0,atol=0)
    # Both SAM passes see identical current targets; stale batch rejected afterwards.
    torch.testing.assert_close(t.get(b['sample_ids'],'cpu'),c.payload['logits'].float(),rtol=0,atol=0)
    with pytest.raises(ValueError):t.get(b['sample_ids'][::-1],'cpu')
    it.close()
    with pytest.raises(ValueError):t.get(b['sample_ids'],'cpu')
    t.fixed_crop=False;s.fixed_crop_distillation['mode']='proxy'
    proxy_loader=FixedCropLoader([b],s,t,c);pi=iter(proxy_loader);proxy=next(pi)
    torch.testing.assert_close(actual['density'],proxy['density'],rtol=0,atol=0)
    torch.testing.assert_close(actual['fixation'],proxy['fixation'],rtol=0,atol=0)
    assert all(a.tobytes()==p.tobytes() for a,p in zip(actual['images'],proxy['images']))
    target_prob=t.get(b['sample_ids'],'cpu').flatten(1).softmax(-1).reshape_as(b['density'])
    expected=warp_density(t.get_raw(b['sample_ids']).flatten(1).softmax(-1).reshape_as(b['density']),BOX,False)
    torch.testing.assert_close(target_prob,expected)
    pi.close()
    assert loader.rng.getstate()==proxy_loader.rng.getstate()
    assert torch.equal(rng,torch.random.get_rng_state())
    torch.testing.assert_close(b['density'],original['density'],rtol=0,atol=0)
    assert all(a.tobytes()==p.tobytes() for a,p in zip(b['images'],original['images']))


def test_empty_fixation_fallback_and_disabled_polish_are_original(tmp_path):
    s,t,c,b=setup(tmp_path)
    b['fixation'][0].zero_();b['fixation'][0,0,0,0]=1
    loader=FixedCropLoader([b],s,t,c);it=iter(loader);out=next(it)
    assert out['images'][0] is b['images'][0]
    assert loader.epoch_fallback_images==1 and loader.epoch_augmented_images==1
    torch.testing.assert_close(out['fixation'][0],b['fixation'][0],rtol=0,atol=0)
    torch.testing.assert_close(t.get(b['sample_ids'],'cpu')[0],t.get_raw(b['sample_ids'])[0],rtol=0,atol=0)
    it.close();loader.enabled=False
    state=loader.rng.getstate();it=iter(loader);out=next(it)
    assert loader.rng.getstate()==state
    assert loader.epoch_augmented_images==loader.epoch_fallback_images==0 and loader.epoch_images==2
    assert all(a is p for a,p in zip(out['images'],b['images']))
    torch.testing.assert_close(t.get(b['sample_ids'],'cpu'),t.get_raw(b['sample_ids']),rtol=0,atol=0)
    it.close()


def test_mixed_views_match_in_both_arms_and_reject_wrong_pixels(tmp_path):
    s,t,c,b=setup(tmp_path,probability=.5)
    actual=FixedCropLoader([b]*30,s,t,c)
    views=[]
    for batch in actual:views.append([im is src for im,src in zip(batch['images'],b['images'])])
    assert 15 < actual.epoch_augmented_images < 45 and actual.epoch_images==60
    s.fixed_crop_distillation=dict(s.fixed_crop_distillation,mode='proxy')
    proxy=FixedCropLoader([b]*30,s,t,c)
    assert views==[[im is src for im,src in zip(batch['images'],b['images'])] for batch in proxy]
    b['images'][0]=b['images'][0].transpose(0)
    with pytest.raises(ValueError,match='pixels differ'):next(iter(proxy))
    b['sample_ids'][0]='val/wrong'
    with pytest.raises(ValueError,match='identity'):next(iter(proxy))


def test_target_bytes_and_annotation_contract(tmp_path,monkeypatch):
    from LightGenV2.tasks.t03_saliency import recheck_aligned,modeling
    s,t,c,b=setup(tmp_path)
    payload=deepcopy(c.payload)
    payload['manifest']['checkpoint_sha256']=s.distillation_teacher_sha256
    monkeypatch.setattr(recheck_aligned,'load_hashed_checkpoint',lambda p:(payload,s.fixed_crop_distillation['cache_sha256']))
    monkeypatch.setattr(modeling,'sha256_file',lambda p:'b'*64)
    records=[SimpleNamespace(sample_id=sid) for sid in b['sample_ids']]
    targets=FixedCropTargets(s,records)
    assert targets.index==t.index and targets.provenance['inference_parameters_added']==0
    monkeypatch.setattr(recheck_aligned,'load_hashed_checkpoint',lambda p:(payload,'c'*64))
    with pytest.raises(ValueError,match='bytes SHA'):FixedCropTargets(s,records)


def test_configs_no_inference_change_and_reject_uncontrolled_combinations(tmp_path):
    a=load_settings(ROOT/'moe_alpha40_fixed_crop_actual.yaml')
    p=load_settings(ROOT/'moe_alpha40_fixed_crop_proxy.yaml')
    old=load_settings(ROOT/'moe_alpha40_extra_control.yaml')
    assert architecture_label(a)==architecture_label(p)==architecture_label(old)
    assert {k:v for k,v in a.fixed_crop_distillation.items() if k!='mode'}=={
        k:v for k,v in p.fixed_crop_distillation.items() if k!='mode'}
    for key in ['student_epochs','staged_polish_start','initialization_checkpoint_sha256','expert_size',
                'pixel_pitch_um','router_backend','top_k','fusion_alpha_min','ema_decay','sam_rho',
                'distillation_initial_weight','distillation_final_weight','electronic_width',
                'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(a,key)==getattr(p,key)==getattr(old,key)
    assert a.student_epochs==40 and a.fixed_crop_distillation['end_epoch']==25
    a.output_dir=tmp_path;save_resolved_config(a)
    import yaml
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['fixed_crop_distillation']==a.fixed_crop_distillation
    for key,value in [('mode','random'),('cache_sha256','short'),('apply_probability',float('nan')),('end_epoch',40)]:
        opts=dict(a.fixed_crop_distillation);opts[key]=value
        with pytest.raises(ValueError):configure(deepcopy(a),opts,ROOT)
    for key,value in [('augmentation_enabled',True),('first_stage_supervision',{'enabled':True}),('fusion_alpha_min',.1)]:
        s=deepcopy(a);setattr(s,key,value)
        with pytest.raises(ValueError):configure(s,s.fixed_crop_distillation,ROOT)
