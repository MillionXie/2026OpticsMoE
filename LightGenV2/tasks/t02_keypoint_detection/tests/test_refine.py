import math
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t02_keypoint_detection.refine import (
    PROFILES, RATES, apply_stage, phase_delta, stage_spec,
)


def test_schedule_complete_and_stage_boundaries():
    for profile in PROFILES:
        for epoch in range(1, 61):
            name, rates = stage_spec(profile, epoch)
            assert name and set(rates) == set(RATES)
            assert all(value >= 0 for value in rates.values())
    for epoch, expected in [(1, 'head_calibration'), (5, 'head_calibration'),
                            (6, 'optics_readout'), (15, 'optics_readout'),
                            (16, 'joint_refinement'), (50, 'joint_refinement'),
                            (51, 'fixed_optics_polish'), (60, 'fixed_optics_polish')]:
        assert stage_spec('staged', epoch)[0] == expected
        assert stage_spec('staged', epoch) == stage_spec('staged_heatmap', epoch)
    with pytest.raises(ValueError):
        stage_spec('staged', 61)


def test_frozen_groups_do_not_move_and_can_thaw():
    parameters = {name: torch.nn.Parameter(torch.ones(2)) for name in RATES}
    opt = torch.optim.AdamW([{'name': k, 'params': [p]} for k, p in parameters.items()], weight_decay=.1)
    for p in parameters.values():
        p.grad = torch.ones_like(p)
    report = apply_stage(opt, stage_spec('staged', 1)[1])
    assert all(p.grad is None for p in parameters.values())
    sum(p.sum() for p in parameters.values() if p.requires_grad).backward()
    opt.step()
    for name, p in parameters.items():
        assert report[name]['trainable_parameters'] == (2 if name == 'pose_head' else 0)
        assert torch.equal(p, torch.ones(2)) == (name != 'pose_head')
    apply_stage(opt, stage_spec('staged', 16)[1])
    assert all(p.requires_grad for p in parameters.values())
    with pytest.raises(ValueError):
        apply_stage(opt, {'pose_head': 1e-4})


def test_phase_delta_tracks_physical_phase():
    core = torch.nn.Module()
    core.raw_phase = torch.nn.Parameter(torch.zeros(2, 2))
    reference = {'raw_phase': torch.zeros(2, 2)}
    model = SimpleNamespace(core=core)
    assert phase_delta(model, reference)['raw_phase']['raw_rms_change'] == 0
    with torch.no_grad():
        core.raw_phase.fill_(1)
    result = phase_delta(model, reference)['raw_phase']
    assert result['raw_rms_change'] == 1
    assert result['physical_phase_rms_change_rad'] == pytest.approx(2*math.pi*(torch.sigmoid(torch.tensor(1.)).item()-.5))
