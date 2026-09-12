from copy import deepcopy
from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t03_saliency.audit_checkpoint import audit_payload, PHASE_SHAPES
from LightGenV2.tasks.t03_saliency.aligned_baseline import AlignedReadout
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.settings import load_settings

TASK = Path(__file__).resolve().parents[1]


def fixture():
    s = load_settings(TASK/'configs/moe_alpha40_reliable_teacher50.yaml')
    head = AlignedReadout()
    core = {k: torch.zeros(shape) for k,shape in PHASE_SHAPES.items()}
    core.update({f'hybrid.block{i}_optical_fusion_logit': torch.tensor(-2.5) for i in (1, 2)})
    p = {'architecture': architecture_label(s), 'core': core, 'saliency_head': head.decoder.state_dict(),
         'fusion_contract': {'minimum': .4, 'maximum': 1.}, 'epoch': 3, 'weight_kind': 'ema'}
    q = {'architecture': 'frozen_qwen24_adapter192_identical_progressive_decoder_v1', 'head': head.state_dict()}
    return s, p, deepcopy(p), q


def test_checkpoint_audit_checks_state_not_accuracy():
    s, candidate, reference, qwen = fixture()
    r = audit_payload(candidate, reference, s, qwen)
    assert r['state_checks_passed'] and r['runtime_and_full_split_recheck_still_required']
    assert r['decoder_parameters'] == 85412
    assert r['optical_parameters_including_router'] == 479364
    assert r['same_qwen_decoder_specification']
    assert all(.4 <= x <= 1 for x in r['alpha'])
    assert all(x == 0 for x in r['phase_rms_change_rad_vs_reference'].values())
    key = next(iter(PHASE_SHAPES))
    candidate['core'][key].add_(.1)
    assert audit_payload(candidate, reference, s)['phase_rms_change_rad_vs_reference'][key] > 0


@pytest.mark.parametrize('failure', ['alpha_contract', 'nan', 'shape', 'extra_key', 'config', 'qwen_head'])
def test_checkpoint_audit_rejects_contract_drift(failure):
    s, p, reference, qwen = fixture()
    key = next(iter(PHASE_SHAPES))
    if failure == 'alpha_contract': p['fusion_contract']['minimum'] = .1
    if failure == 'nan': p['core'][key][0,0] = float('nan')
    if failure == 'shape': p['core'][key] = p['core'][key].flatten()
    if failure == 'extra_key': p['core']['unexpected_branch.weight'] = torch.zeros(2)
    if failure == 'config': s.top_k = 4
    if failure == 'qwen_head': del qwen['head'][next(k for k in qwen['head'] if k.startswith('decoder.'))]
    with pytest.raises(ValueError): audit_payload(p, reference, s, qwen)
