from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t03_saliency.plateau import PlateauController


def test_grace_reductions_stop_and_best_protected():
    p = PlateauController()
    assert p.observe(0, .85812) == "improved"
    for epoch in (1, 2, 4):
        assert p.observe(epoch, .857) == "grace"
    for epoch in (6, 8, 12, 14, 18, 20):
        # Third failed evaluation in each window triggers a reduction/stop.
        assert p.observe(epoch, .857) == "wait"
        if epoch in (8, 14, 20):
            assert p.observe(epoch + 2, .857) == ("stop" if epoch == 20 else "reduce_lr")
    assert p.multiplier == .25 and p.stopped and p.best == .85812


def test_real_improvement_resets_patience_but_keeps_reductions():
    p = PlateauController(patience=1, min_epoch=1)
    p.observe(0, .85)
    assert p.observe(1, .849) == "reduce_lr"
    assert p.observe(2, .851) == "improved"
    assert p.bad_tests == 0 and p.multiplier == .5
    assert p.observe(3, .85101) == "reduce_lr"


def test_scaling_after_closed_form_schedule_does_not_compound():
    p = PlateauController(patience=1, min_epoch=1)
    p.observe(0, .85)
    p.observe(1, .84)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.001}, {"lr": 0.0}])
    for _ in range(3):
        optimizer.param_groups[0]["lr"] = .001  # staged_epoch recomputes this
        p.scale_epoch_rates(optimizer)
        assert optimizer.param_groups[0]["lr"] == .0005
        assert optimizer.param_groups[1]["lr"] == 0


@pytest.mark.parametrize("options", [{"patience": 0}, {"factor": 1}, {"min_delta": -1}, {"min_delta": float('nan')}])
def test_reject_bad_options(options):
    with pytest.raises(ValueError):
        PlateauController(**options)


def test_nonfinite_metric_fails_instead_of_selecting_weights():
    with pytest.raises(ValueError):
        PlateauController().observe(1, float('nan'))


def test_profiles_keep_inference_contract():
    from pathlib import Path
    from LightGenV2.tasks.t03_saliency.settings import load_settings
    from LightGenV2.tasks.t03_saliency.modeling import architecture_label
    root = Path(__file__).resolve().parents[1] / 'configs'
    base = load_settings(root / 'moe_alpha40_rfstage_control.yaml')
    for name in ('keepkd', 'releasekd', 'releasekd_cc'):
        s = load_settings(root / f'moe_alpha40_adaptive_{name}.yaml')
        assert s.adaptive_plateau_enabled and s.test_interval_epochs == 2
        assert s.initialization_checkpoint_sha256 == base.initialization_checkpoint_sha256
        assert architecture_label(s) == architecture_label(base)
        assert s.top_k == 2 and s.router_backend == 'optical' and s.fusion_alpha_min == .4
        assert s.language_optical_phase_zero_order_intensity_min == .2
        assert s.language_optical_phase_zero_order_intensity_max == .3
        assert not s.reset_fusion_on_warmstart
        assert not s.augmentation_enabled
        assert s.distillation_final_weight == (.6 if name == 'keepkd' else 0)
        assert s.cc_weight == (3.0 if name == 'releasekd_cc' else base.cc_weight)
