import copy

import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import RetrievalHead
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES, prepare_capacity_payload
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import expand_retrieval_head

PROTOCOL = 'abo200_enrolled_sku_hash8train4query_v1'


def fixture():
    torch.manual_seed(5)
    old = RetrievalHead().eval()
    state = {'readout.' + k: v.clone() for k, v in old.state_dict().items()}
    state['vision.optics.global_phase'] = torch.randn(8, 8)
    state['frontend.marker'] = torch.randn(3)
    source = dict(metadata={}, state_dict=state)
    converted, audit = prepare_capacity_payload(source, PROFILES['sku_spatial_readout'], PROTOCOL)
    new = RetrievalHead('spatial2x2_64').eval()
    new.load_state_dict({k.removeprefix('readout.'): v for k, v in converted['state_dict'].items() if k.startswith('readout.')})
    mask = torch.zeros(2, 77, dtype=torch.bool); mask[:, 11:60] = True
    return old, new, source, converted, audit, mask


def test_spatial_conversion_preserves_initial_function_and_optical_weights():
    old, new, source, converted, audit, mask = fixture()
    x = torch.randn(2, 77, 192, requires_grad=True)
    torch.testing.assert_close(new(x, mask), old(x), rtol=2e-6, atol=2e-7)
    assert audit['extra_parameters'] == 49152
    assert sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()) == 49152
    for name, value in source['state_dict'].items():
        if name != 'readout.projection.weight': assert torch.equal(value, converted['state_dict'][name])
    assert source['state_dict']['readout.projection.weight'].shape == (64,384)
    assert source['metadata'] == {} and converted['metadata']['retrieval_head'] == 'spatial2x2_64'
    loss = (new(x, mask) - torch.randn(2,64)).square().mean(); loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(new.projection.weight.grad).all() and new.projection.weight.grad[:,384:].abs().sum() > 0
    assert expand_retrieval_head(converted, 'spatial2x2_64') is converted


def test_spatial_cells_see_position_which_global_pool_does_not():
    old, new, _, _, _, mask = fixture()
    x = torch.randn(2,77,192); swapped = x.clone()
    swapped[:,11:60] = x[:,11:60].flip(1)
    torch.testing.assert_close(old(x), old(swapped), rtol=1e-5, atol=1e-6)
    with torch.no_grad(): new.projection.weight[:,384:].normal_(0,.02)
    assert (new(x,mask) - new(swapped,mask)).abs().max() > .01


@pytest.mark.parametrize('kind', ['none','count','dtype','shape'])
def test_spatial_head_rejects_wrong_image_token_layout(kind):
    _, new, _, _, _, mask = fixture(); x = torch.randn(2,77,192)
    if kind == 'none': mask = None
    if kind == 'count': mask[:,11] = False
    if kind == 'dtype': mask = mask.long()
    if kind == 'shape': mask = mask[:1]
    with pytest.raises(ValueError): new(x,mask)


def test_spatial_profile_only_changes_readout_and_refuses_mixed_conversion():
    _, _, source, converted, _, _ = fixture()
    p = copy.deepcopy(PROFILES['sku_spatial_readout'])
    assert p.pop('head_expansion') == 'spatial2x2_64' and p == PROFILES['sku_capacity_control']
    for protocol, fresh in [('old',False),(PROTOCOL,True)]:
        with pytest.raises(ValueError): prepare_capacity_payload(source,PROFILES['sku_spatial_readout'],protocol,fresh)
    for p in (dict(PROFILES['sku_spatial_readout'], optical_only=True),
              dict(PROFILES['sku_spatial_readout'], electronic_expansion={'kernels':{}})):
        with pytest.raises(ValueError): prepare_capacity_payload(source,p,PROTOCOL)
    with pytest.raises(ValueError): prepare_capacity_payload(converted,PROFILES['sku_spatial_readout'],PROTOCOL)
