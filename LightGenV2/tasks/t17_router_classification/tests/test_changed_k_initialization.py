import pytest
from LightGenV2.tasks.t17_router_classification.train_canonical_head_TEST import initialization_profile


def parent(k=1,dataset='kather2016'):
    return dict(metadata=dict(profile=dict(top_k=k,dataset=dataset,architecture='LG',main_layers=2)))


def test_no_requested_change_preserves_original_profile_object():
    value=parent();assert initialization_profile(value,{}) is value['metadata']['profile']


def test_explicit_adjacent_change_does_not_mutate_parent():
    value=parent();actual=initialization_profile(value,dict(initialize_top_k=2,changed_k_initialization=True))
    assert actual['top_k']==2 and value['metadata']['profile']['top_k']==1


@pytest.mark.parametrize('value,cfg',[(parent(),dict(initialize_top_k=2)),(parent(dataset='eurosat_original_f'),dict(initialize_top_k=2,changed_k_initialization=True)),(parent(2),dict(initialize_top_k=1,changed_k_initialization=True))])
def test_implicit_wrong_domain_or_decreasing_changes_rejected(value,cfg):
    with pytest.raises(ValueError):initialization_profile(value,cfg)
