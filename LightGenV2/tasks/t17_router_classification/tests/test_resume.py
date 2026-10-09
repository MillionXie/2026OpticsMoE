import numpy as np
import pytest
from LightGenV2.tasks.t17_router_classification.train import shuffle_rng,validate_resume

def test_legacy_and_saved_shuffle_continuation_match_uninterrupted():
    uninterrupted=np.random.default_rng(17)
    for _ in range(18):uninterrupted.permutation(5026)
    reconstructed=shuffle_rng(17,18,5026)
    restored=shuffle_rng(0,0,1,uninterrupted.bit_generator.state)
    for _ in range(3):
        expected=uninterrupted.permutation(5026)
        assert np.array_equal(expected,reconstructed.permutation(5026))
        assert np.array_equal(expected,restored.permutation(5026))

@pytest.mark.parametrize('key',['architecture','profile','data_sha256','manifest_sha256',
                              'batch','seed','router_features','router_lr','optical_contract'])
def test_resume_rejects_changed_scientific_contract(key):
    a={key:'original'};b={key:'changed'}
    with pytest.raises(ValueError,match=key):validate_resume(a,b)
