import copy
import json
import random
from dataclasses import replace
from pathlib import Path
import pytest
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.domain_data import combine_training, epoch_batches
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config


def fixture_samples():
    def rows(prefix, count, split):
        return [Sample(f'{prefix}{c}_{p}_{v}',f'{prefix}{c}_{p}',c,f'cat{c}',split,Path('unused'))
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
