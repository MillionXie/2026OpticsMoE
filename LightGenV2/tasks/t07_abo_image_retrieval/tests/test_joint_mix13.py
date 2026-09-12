"""Reduce repeated target-product exposure without changing dataset membership."""
import random
from types import SimpleNamespace
from collections import Counter
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.domain_data import epoch_batches
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config,PINNED_TEACHER_PROFILES
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue import PINNED_TEACHER_PROFILES as QUEUE


def test_profile_only_changes_original_external_training_quota():
    old=overlay_config({},'domain_distill_joint_restart')
    new=overlay_config({},'domain_distill_joint_mix13')
    assert old.get('domain_target_products_per_class',2)==2
    assert new.pop('domain_target_products_per_class')==1
    for cfg in (old,new):cfg.pop('protocol')
    assert old==new
    assert 'domain_distill_joint_mix13' in PINNED_TEACHER_PROFILES and 'domain_distill_joint_mix13' in QUEUE


def test_all_original_products_and_external_products_are_seen_with_exact_quotas():
    samples=[]
    for domain,count,views in [('target',12,12),('external',250,2)]:
        for c in range(10):
            for p in range(count):
                for v in range(views):
                    samples.append(SimpleNamespace(category_id=c,product_id=f'{domain}-{c}-{p}',sample_id=f'{domain}-{c}-{p}-{v}'))
    target_count=1440
    before=[s.sample_id for s in samples]
    phase,batches,active=epoch_batches(samples,target_count,'mixed',1,0,128,random.Random(42),1)
    assert phase=='mixed' and len(batches)==128 and active==list(range(len(samples)))
    assert before==[s.sample_id for s in samples]
    seen=set();draws=Counter()
    for batch in batches:
        assert len(batch)==40 and len({samples[i].product_id for i in batch})==40
        for c in range(10):
            assert sum(i<target_count and samples[i].category_id==c for i in batch)==1
            assert sum(i>=target_count and samples[i].category_id==c for i in batch)==3
        seen.update(samples[i].product_id for i in batch)
        draws.update('target' if i<target_count else 'external' for i in batch)
    assert seen=={s.product_id for s in samples}
    assert draws=={'target':1280,'external':3840}
    assert sum(i<target_count for batch in batches for i in batch)==1280<target_count
    # Do not claim every original IMAGE was visited in one epoch.
    _,again,_=epoch_batches(samples,target_count,'mixed',1,0,128,random.Random(42),1)
    assert again==batches
