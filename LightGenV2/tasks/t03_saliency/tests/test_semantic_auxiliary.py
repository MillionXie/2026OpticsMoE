import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency.semantic_auxiliary import SemanticAuxiliary, box_coverage_targets
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label

TASK=Path(__file__).resolve().parents[1]


def make_fixture(tmp_path):
    ids=['unlabeled/coco2017/000000000010','unlabeled/coco2017/000000000020']
    raw={'schema_version':1,'purpose':'training_only_auxiliary_object_presence_not_saliency_ground_truth',
         'image_manifest_sha256':'a'*64,'train_overlap':0,'test_overlap':0,'sample_count':2,
         'additional_human_semantic_supervision':True,'no_saliency_or_fixation_targets_created':True,
         'annotation_sha256':'b'*64,'categories':[{'id':2*i+1,'name':str(i)} for i in range(80)],
         'category_positive_image_counts':[1]+[0]*79,
         'records':[{'sample_id':ids[0],'image_id':10,'positive_category_ids':[1]},
                    {'sample_id':ids[1],'image_id':20,'positive_category_ids':[]}]}
    path=tmp_path/'semantic.json';path.write_text(json.dumps(raw))
    s=SimpleNamespace(semantic_targets=path,semantic_targets_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      electronic_width=192,random_seed=42)
    dataset=SimpleNamespace(sample_ids=ids,manifest_sha256='a'*64)
    return s,dataset,raw


def test_auxiliary_is_small_rng_neutral_and_backpropagates_to_fused_features(tmp_path):
    s,d,_=make_fixture(tmp_path)
    rng=torch.get_rng_state().clone();aux=SemanticAuxiliary(s,d)
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in aux.parameters())==15440
    assert set(aux.state_dict())=={'head.weight','head.bias'}
    assert aux.provenance['inference_parameters_added']==0
    groups=[torch.randn(196,192,requires_grad=True) for _ in d.sample_ids]
    before=aux.head.weight.detach().clone()
    optimizer=torch.optim.AdamW(aux.parameters(),lr=.001)
    loss=aux(groups,d.sample_ids);loss.backward();optimizer.step()
    assert torch.isfinite(loss) and all(g.grad is not None and g.grad.norm()>0 for g in groups)
    assert not torch.equal(before,aux.head.weight)
    assert torch.isfinite(aux.positive_weight).all() and aux.positive_weight.min()>=1 and aux.positive_weight.max()<=10
    # The auxiliary module does not own or modify any inference parameters.
    assert all(name.startswith('head.') for name,_ in aux.named_parameters())


@pytest.mark.parametrize('failure',['sha','order','overlap','counts'])
def test_semantic_cache_is_strict(tmp_path,failure):
    s,d,raw=make_fixture(tmp_path)
    if failure=='sha':s.semantic_targets_sha256='0'*64
    elif failure=='order':d.sample_ids.reverse()
    else:
        if failure=='overlap':raw['test_overlap']=1
        if failure=='counts':raw['category_positive_image_counts'][0]=0
        s.semantic_targets.write_text(json.dumps(raw))
        s.semantic_targets_sha256=hashlib.sha256(s.semantic_targets.read_bytes()).hexdigest()
    with pytest.raises(ValueError):SemanticAuxiliary(s,d)


def test_published_profile_keeps_inference_geometry_gt_and_data_budget():
    base=load_settings(TASK/'configs/moe_alpha40_extra_early_coco20k.yaml')
    s=load_settings(TASK/'configs/moe_alpha40_extra_semantic.yaml')
    assert architecture_label(s)==architecture_label(base)
    assert s.semantic_weight==.5 and base.semantic_weight==0
    assert s.semantic_learning_rate==.001 and s.electronic_width==192
    for key in ('student_epochs','initialization_checkpoint_sha256','sam_rho','ema_decay',
                'student_learning_rate','phase_learning_rate','router_learning_rate',
                'kl_weight','cc_weight','sim_weight','nss_weight','unlabeled_cache_sha256',
                'unlabeled_weight','fusion_alpha_min','top_k','active_size','expert_size'):
        assert getattr(s,key)==getattr(base,key),key
    assert s.fusion_alpha_min==.4 and s.router_backend=='optical' and s.top_k==2
    assert s.language_optical_phase_zero_order_intensity_min==.2
    assert s.language_optical_phase_zero_order_intensity_max==.3


def test_semantic_auxiliary_requires_pinned_extra_stream(tmp_path):
    p=tmp_path/'LightGenV2/tasks/t03_saliency/configs/invalid.yaml';p.parent.mkdir(parents=True)
    p.write_text(f'base_config: {(TASK/"configs/moe_alpha40_extra_semantic.yaml").as_posix()}\n'
                 'unlabeled_distillation:\n  weight: 0\n')
    with pytest.raises(ValueError,match='extra-image stream'):load_settings(p)


def region_fixture(tmp_path):
    s,d,raw=make_fixture(tmp_path)
    s.semantic_mode='box_coverage'
    raw.update(purpose='training_only_auxiliary_object_regions_not_saliency_ground_truth',
               additional_human_box_supervision=True,spatial_box_count=1,
               image_resize_contract='direct anisotropic RGB BICUBIC resize to 224x224; no crop/EXIF transpose')
    raw['records'][0]['boxes_xyxy_unit']=[{'category_id':1,'xyxy':[0,0,.5,.5]}]
    raw['records'][1]['boxes_xyxy_unit']=[]
    s.semantic_targets.write_text(json.dumps(raw))
    s.semantic_targets_sha256=hashlib.sha256(s.semantic_targets.read_bytes()).hexdigest()
    return s,d,raw


def test_box_coverage_keeps_fractional_cells_and_does_not_double_count():
    boxes=[{'category_id':1,'xyxy':[0,0,1/28,1/14]}]*2
    labels=box_coverage_targets(boxes,{1:0}).reshape(14,14,80)
    assert labels[0,0,0]==.5 and labels.sum()==.5
    assert box_coverage_targets([],{1:0}).sum()==0


@pytest.mark.parametrize('coords',[[0,0,0,1],[0,0,2,1],[0,0,float('nan'),1],None])
def test_invalid_spatial_boxes_rejected(coords):
    with pytest.raises(ValueError):box_coverage_targets([{'category_id':1,'xyxy':coords}],{1:0})


def test_spatial_auxiliary_restores_qwen_layout_and_keeps_inference_unchanged(tmp_path):
    s,d,_=region_fixture(tmp_path)
    rng=torch.get_rng_state().clone();aux=SemanticAuxiliary(s,d)
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in aux.parameters())==15440
    assert set(aux.state_dict())=={'head.weight','head.bias'}
    assert aux.provenance['positive_target_mass'][0]==49
    assert aux.provenance['positive_image_counts'][0]==1
    assert aux.provenance['target_observations_per_category']==392
    grid=torch.randn(2,14,14,192)
    packed=grid.reshape(2,7,2,7,2,192).permute(0,1,3,2,4,5).reshape(2,196,192).requires_grad_()
    captured=[]
    handle=aux.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    loss=aux(list(packed),d.sample_ids);handle.remove()
    expected=torch.nn.functional.layer_norm(grid.reshape(2,196,192),(192,))
    torch.testing.assert_close(captured[0],expected)
    loss.backward()
    assert torch.isfinite(loss) and packed.grad.norm()>0 and aux.head.weight.grad.norm()>0
    assert torch.isfinite(aux.positive_weight).all()
    # No cross-image pooling: joint BCE is the mean of individually evaluated BCEs.
    torch.testing.assert_close(loss,torch.stack([aux([packed[i]],[key]) for i,key in enumerate(d.sample_ids)]).mean())
    with pytest.raises(ValueError,match='196'):aux([torch.randn(195,192)],d.sample_ids[:1])


def test_regions_cannot_be_accidentally_loaded_as_image_presence(tmp_path):
    s,d,_=region_fixture(tmp_path);s.semantic_mode='image_presence'
    with pytest.raises(ValueError,match='provenance'):SemanticAuxiliary(s,d)


def test_region_profile_only_changes_training_semantic_target():
    base=load_settings(TASK/'configs/moe_alpha40_extra_semantic.yaml')
    new=load_settings(TASK/'configs/moe_alpha40_extra_regions.yaml')
    assert architecture_label(base)==architecture_label(new)
    allowed={'semantic_mode','semantic_targets','semantic_targets_sha256','output_dir','config_path','config'}
    differences={key for key in vars(base) if getattr(base,key)!=getattr(new,key)}
    assert differences<=allowed,differences
    assert new.semantic_mode=='box_coverage' and new.semantic_weight==.5
