import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency.semantic_auxiliary import SemanticAuxiliary
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
