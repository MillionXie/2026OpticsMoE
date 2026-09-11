import copy
import json
import random
from dataclasses import replace
from pathlib import Path
import pytest
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.domain_data import combine_training, epoch_batches, paired_view_indices, view_consistency_loss
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config


def fixture_samples():
    def rows(prefix, count, split):
        return [Sample(f'{prefix}{c}_{p}_{v}',f'{prefix}{c}_{p}',c,f'cat{c}',split,Path(f'{prefix}{c}_{p}_{v}.png'))
                for c in range(2) for p in range(count) for v in range(2)]
    return rows('t',4,'train')+rows('q',2,'test'), rows('e',7,'pretrain')


def test_combine_does_not_change_target_or_evaluation():
    target,external=fixture_samples();before=copy.deepcopy(target)
    combined,n=combine_training(target,external)
    assert target==before and n==16 and len(combined)==44
    assert all(s.split=='train' for s in combined)
    assert not {s.product_id for s in combined}&{s.product_id for s in target if s.split=='test'}
    assert all(s.split=='pretrain' for s in external)


def test_combine_rejects_overlap_and_semantic_mismatch():
    target,external=fixture_samples()
    with pytest.raises(ValueError):combine_training(target,[replace(external[0],product_id=target[-1].product_id)]+external[1:])
    with pytest.raises(ValueError):combine_training(target,[replace(external[0],category_name='wrong')]+external[1:])


@pytest.mark.parametrize('mode,epoch,phase',[('mixed',1,'mixed'),('curriculum',1,'external'),('curriculum',11,'mixed'),('target_control',1,'target')])
def test_balanced_coverage_and_curriculum(mode,epoch,phase):
    target,external=fixture_samples();samples,n=combine_training(target,external)
    actual,batches,active=epoch_batches(samples,n,mode,epoch,10,1,random.Random(42))
    assert actual==phase
    visited={i for batch in batches for i in batch}
    assert {samples[i].product_id for i in visited}=={samples[i].product_id for i in active}
    for batch in batches:
        assert len(batch)==8 and len({samples[i].product_id for i in batch})==8
        assert sum(i<n for i in batch)=={'mixed':4,'external':0,'target':8}[phase]


def test_profiles_only_change_domain_schedule():
    configs=[overlay_config({},'domain_'+n) for n in ['mixed','curriculum','target_control']]
    for cfg in configs:
        cfg.pop('domain_mode')
        assert cfg['adapt']['epochs']==40 and cfg['adapt']['readout_polish_epochs']==0
    assert configs[0]==configs[1]==configs[2]


def test_old_profiles_still_resolve():
    assert overlay_config({},'regularized_phase05')['phase_dropout']['expert_global_probability']==.05
    assert overlay_config({},'preserve_adam')['input_preprocessing']=='contain_white'


def test_alternate_views_are_same_product_different_image():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer import make_groups
    target,external=fixture_samples();samples,n=combine_training(target,external)
    ids=list(range(len(samples)));pairs=paired_view_indices(samples,make_groups(samples),ids,random.Random(5))
    for i,j in zip(ids,pairs):
        assert i!=j and samples[i].product_id==samples[j].product_id
        assert samples[i].image_path!=samples[j].image_path and samples[j].split=='train'


def test_pairing_rejects_single_view_instead_of_using_another_product():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer import make_groups
    target,_=fixture_samples();one=[target[0]]
    with pytest.raises(ValueError):paired_view_indices(one,make_groups(one),[0],random.Random(1))


def test_consistency_backpropagates_both_views():
    import torch
    a=torch.tensor([[1.,0.,0.]],requires_grad=True)
    b=torch.tensor([[0.,1.,0.]],requires_grad=True)
    loss=view_consistency_loss(a,b);loss.backward()
    assert loss.item()==pytest.approx(1.)
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert view_consistency_loss(a,a).item()==pytest.approx(0.)
    with pytest.raises(ValueError):view_consistency_loss(a,b.repeat(2,1))


def test_refinement_config_changes_only_controlled_factors():
    control=overlay_config({},'domain_refine_control')
    wide=overlay_config({},'domain_refine_wide')
    views=overlay_config({},'domain_refine_views')
    assert control['adapt']['epochs']==24 and wide['adapt']['steps']==128
    assert control.pop('expected_pool_products_per_category')==100
    assert wide.pop('expected_pool_products_per_category')==250
    assert views.pop('expected_pool_products_per_category')==250
    assert control==wide
    assert views['view_consistency_weight']==.15
    views['view_consistency_weight']=0.
    assert views==wide


def test_mixed_quota_preserves_coverage_and_default_sequence():
    target,external=fixture_samples();samples,n=combine_training(target,external)
    old=epoch_batches(samples,n,'mixed',1,10,1,random.Random(42))
    explicit=epoch_batches(samples,n,'mixed',1,10,1,random.Random(42),2)
    assert old==explicit
    phase,batches,active=epoch_batches(samples,n,'mixed',1,10,1,random.Random(42),1)
    assert phase=='mixed' and len(batches)==4
    visited={samples[i].product_id for b in batches for i in b}
    assert visited=={samples[i].product_id for i in active}
    for batch in batches:
        assert len({samples[i].product_id for i in batch})==8
        assert sum(i<n for i in batch)==2
        for c in (0,1):
            assert sum(i<n and samples[i].category_id==c for i in batch)==1
            assert sum(i>=n and samples[i].category_id==c for i in batch)==3
    for q in (0,4,1.5,True):
        with pytest.raises(ValueError):epoch_batches(samples,n,'mixed',1,10,1,random.Random(42),q)
    cfg=overlay_config({},'domain_refine_pool500_mix13')
    assert cfg['domain_target_products_per_class']==1 and cfg['expected_pool_products_per_category']==500
    assert not cfg.get('relation_teacher_weight',0) and cfg['view_consistency_weight']==0
