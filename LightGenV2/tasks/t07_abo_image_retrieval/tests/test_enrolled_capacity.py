import copy

import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import Residual
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES, prepare_capacity_payload

PROTOCOL = 'abo200_enrolled_sku_hash8train4query_v1'


def payload():
    torch.manual_seed(42)
    state = {}
    for mode in ('vision', 'language'):
        for i in (0, 1):
            state.update({f'{mode}.blocks.{i}.' + k: v
                          for k, v in Residual(mode == 'vision').state_dict().items()})
        state[mode + '.optics.global_phase'] = torch.randn(478, 478)
        state[mode + '.block1_optical_fusion_logit'] = torch.tensor(-.2)
    state['frontend.marker'] = torch.randn(3, 4)
    state['readout.marker'] = torch.randn(64, 384)
    return dict(metadata={}, state_dict=state)


@pytest.mark.parametrize('vision', [True, False])
def test_capacity_conversion_preserves_function_optics_and_source(vision):
    source = payload()
    snapshot = copy.deepcopy(source)
    converted, audit = prepare_capacity_payload(source, PROFILES['sku_conv_teacher'], PROTOCOL)
    assert audit['extra_parameters'] == 607488
    assert len(audit['changed_tensor_shapes']) == 16
    mode = 'vision' if vision else 'language'
    prefix = mode + '.blocks.0.'
    old, new = Residual(vision), Residual(vision, kernel_size=7, mlp_width=768)
    for model, state in ((old, source['state_dict']), (new, converted['state_dict'])):
        model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        model.eval()
    x = torch.randn(2, 196 if vision else 77, 192)
    with torch.no_grad():
        torch.testing.assert_close(old(x), new(x), rtol=3e-6, atol=3e-6)
    for name, value in source['state_dict'].items():
        assert torch.equal(value, snapshot['state_dict'][name])
        if '.blocks.' not in name:
            assert converted['state_dict'][name] is value
    assert source['metadata'] == snapshot['metadata']


def test_capacity_pair_differs_only_in_expansion_and_guards_protocol():
    teacher = dict(PROFILES['sku_conv_teacher'])
    teacher.pop('electronic_expansion')
    assert teacher == PROFILES['sku_capacity_control']
    source = payload()
    result, audit = prepare_capacity_payload(source, teacher, PROTOCOL)
    assert result is source and audit == dict(expanded=False, extra_parameters=0)
    for protocol, fresh in [('old_abo', False), (PROTOCOL, True)]:
        with pytest.raises(ValueError):
            prepare_capacity_payload(source, PROFILES['sku_conv_teacher'], protocol, fresh)
