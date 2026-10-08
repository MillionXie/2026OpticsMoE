import math
import json
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t02_keypoint_detection.refine import (
    PROFILES, RATES, apply_stage, phase_delta, stage_spec, profile_epochs, polish_config,
)


def test_schedule_complete_and_stage_boundaries():
    for profile in PROFILES:
        for epoch in range(1, profile_epochs(profile)+1):
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


def test_weights_only_continuation_identity_and_guards(tmp_path):
    from LightGenV2.tasks.t02_keypoint_detection.refine import continuation_metadata, HIGH_SOURCE_SHA
    args = SimpleNamespace(profile='bounded_staged', smoke=False, seed=42,
                           batch_size=12, workers=0, data_root=tmp_path, cache_dir=tmp_path)
    manifest = {'args': {k:str(v) if k in ('data_root','cache_dir') else v for k,v in vars(args).items()},
                'source_sha256':HIGH_SOURCE_SHA, 'physical_amplitude_mode':'tanh05_uint8'}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'training_history.json').write_text(json.dumps([{'epoch':i} for i in range(1,21)]))
    for name in ('best_checkpoint.pt','last_checkpoint.pt','pose_protocol_split.csv','source_anchor_test.json'):
        (tmp_path/name).write_text('fixture')
    metadata=continuation_metadata(tmp_path,args)
    assert metadata['completed_epoch']==20 and not metadata['exact_resume']
    assert metadata['optimizer_reset'] and metadata['ema_reset_to_live_last']
    args.seed=43
    with pytest.raises(ValueError,match='seed'): continuation_metadata(tmp_path,args)
    args.seed=42
    (tmp_path/'training_history.json').write_text('[{"epoch":2}]')
    with pytest.raises(ValueError,match='history'): continuation_metadata(tmp_path,args)
    (tmp_path/'final_report.json').write_text('{}')
    with pytest.raises(ValueError,match='completed'): continuation_metadata(tmp_path,args)


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


@pytest.mark.parametrize('profile,lower,initial', [('alpha50',.5,.55), ('alpha40',.4,.42)])
def test_high_alpha_contract_reset_and_bounds(profile,lower,initial):
    from pathlib import Path
    from LightGenV2.tasks.t02_keypoint_detection.settings import load_settings
    from LightGenV2.tasks.t02_keypoint_detection.modeling import architecture_label
    from LightGenV2.tasks.t02_keypoint_detection.refine import checked_fusion
    from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import _ScaleMatchedFusionMixin
    configs = Path(__file__).resolve().parents[1]/'configs'
    high = load_settings(configs/f'moe_{profile}.yaml')
    old = load_settings(configs/'moe_optical_router_scale_matched_dc20_no_shift_warmstart.yaml')
    assert architecture_label(high) != architecture_label(old)
    assert high.fusion_alpha_min == lower and high.coordinate_loss_weight == 0

    class Fusion(_ScaleMatchedFusionMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block1_optical_fusion_logit = torch.nn.Parameter(torch.tensor(-1.88825))
            self.block2_optical_fusion_logit = torch.nn.Parameter(torch.tensor(-2.6356))
            self._configure_balanced_fusion(high)
    fusion = Fusion()
    model = SimpleNamespace(core=SimpleNamespace(hybrid=fusion))
    fusion.reset_fusion_logits(initial)
    assert list(checked_fusion(model,high).values()) == pytest.approx([initial,initial])
    for raw in (-1e6,-10.,0.,10.,1e6):
        with torch.no_grad():
            for p in fusion.parameters(): p.fill_(raw)
        assert all(lower <= v <= .950001 for v in checked_fusion(model,high).values())
    assert stage_spec('alpha50',1)[1]['electronic'] == 0
    assert stage_spec('alpha50',1)[1]['feature_phase'] > 0
    assert stage_spec('alpha50',11)[1]['electronic'] > 0
    assert stage_spec('alpha40',1)[1]['feature_phase'] == 0
    assert stage_spec('alpha40',4)[1]['feature_phase'] > 0
    assert stage_spec('alpha40',4)[1]['electronic'] > 0


def test_polish_keeps_architecture_and_smaller_steps():
    from pathlib import Path
    from LightGenV2.tasks.t02_keypoint_detection.settings import load_settings
    from LightGenV2.tasks.t02_keypoint_detection.modeling import architecture_label
    configs = Path(__file__).resolve().parents[1]/'configs'
    original = load_settings(configs/'moe_alpha40.yaml')
    polish = load_settings(configs/'moe_alpha40_polish.yaml')
    assert architecture_label(original) == architecture_label(polish)
    assert polish.fusion_alpha_min == .4 and polish.coordinate_loss_weight == 0
    assert profile_epochs('alpha40_polish') == 40
    for k,lr in stage_spec('alpha40_polish',1)[1].items():
        assert 0 < lr < stage_spec('alpha40',4)[1][k]
    assert stage_spec('alpha40_polish',30)[1]['feature_phase'] > 0
    assert stage_spec('alpha40_polish',31)[1]['feature_phase'] == 0
    assert len(polish_config()['source_sha256']) == 64
    with pytest.raises(ValueError): stage_spec('alpha40_polish',41)
