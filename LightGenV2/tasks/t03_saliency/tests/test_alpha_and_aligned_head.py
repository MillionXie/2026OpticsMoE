from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency.aligned_baseline import AlignedReadout
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label, initialize_student
from LightGenV2.tasks.t03_saliency.training import staged_epoch, phase_change_report
from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import _range_gate, _ScaleMatchedFusionMixin

TASK = Path(__file__).resolve().parents[1]


def test_refinement_rates_and_unchanged_architecture():
    previous = load_settings(TASK / "configs/moe_staged_alpha_ge040.yaml")
    assert previous.learning_rate_source == "legacy_optimization"
    assert previous.phase_learning_rate == pytest.approx(1e-4)
    names = ("control", "reheat", "cc", "weakaug")
    configs = [load_settings(TASK / f"configs/moe_alpha40_refine_{n}.yaml") for n in names]
    for s in configs:
        assert s.learning_rate_source == "task_training"
        assert s.student_learning_rate == pytest.approx(3e-5)
        assert s.router_learning_rate == pytest.approx(2e-4)
        assert not s.reset_fusion_on_warmstart
        assert s.fusion_alpha_min == .4 and s.top_k == 2
        assert architecture_label(s) == architecture_label(previous)
        assert s.language_optical_zero_order_enabled
        assert s.language_optical_phase_zero_order_intensity_min == .2
        assert s.language_optical_phase_zero_order_intensity_max == .3
        assert s.initialization_checkpoint == configs[0].initialization_checkpoint
        assert s.map_kd_weight == 0
    assert [s.phase_learning_rate for s in configs] == pytest.approx([1e-4, .003, .003, .003])
    assert [s.cc_weight for s in configs] == [.5, .5, 1.5, 1.5]
    assert [s.crop_scale_min for s in configs] == [.90, .90, .90, .98]


def test_aligned_head_counts_and_forward():
    head = AlignedReadout()
    assert head.parameter_audit() == {"adapter": 197184, "decoder": 85412, "total": 282596}
    with torch.no_grad():
        assert head(torch.randn(1, 1024, 14, 14)).shape == (1, 1, 224, 224)


def test_phase_movement_report(tmp_path):
    path = tmp_path / "phase.pt"
    torch.save({"core": {"phase.raw_phase": torch.zeros(4)}}, path)
    s = SimpleNamespace(initialization_checkpoint=path)
    report = phase_change_report({"core": {"phase.raw_phase": torch.zeros(4)}}, s)
    assert report["phase.raw_phase"]["circular_phase_rms_rad"] == 0
    report = phase_change_report({"core": {"phase.raw_phase": torch.ones(4)}}, s)
    assert report["phase.raw_phase"]["fraction_above_001_rad"] == 1


def test_alpha_pair_changes_only_interval_and_run_identity():
    free = load_settings(TASK / "configs/moe_staged_alpha_free.yaml")
    bounded = load_settings(TASK / "configs/moe_staged_alpha_ge040.yaml")
    assert free.fusion_alpha_min == 0
    assert bounded.fusion_alpha_min == .4
    assert free.fusion_alpha_max == bounded.fusion_alpha_max == 1
    assert free.fusion_alpha_initial == bounded.fusion_alpha_initial == .45
    assert free.initialization_checkpoint == bounded.initialization_checkpoint
    assert free.reset_fusion_on_warmstart and bounded.reset_fusion_on_warmstart
    assert architecture_label(free) != architecture_label(bounded)
    values = _range_gate(torch.linspace(-100, 100, 1000), .4, 1.)
    assert values.min() >= .4 and values.max() <= 1
    assert bounded.router_backend == "optical" and bounded.top_k == 2
    assert bounded.ccd_normalization == "mean_only"


def test_stages_have_reproducible_noncompounding_lrs():
    s = load_settings(TASK / "configs/moe_staged_alpha_ge040.yaml")
    opt = SimpleNamespace(param_groups=[{"name": "electronic", "lr": .001}, {"name": "feature_phase", "lr": .01}])
    assert staged_epoch(opt, s, 1)["lr_electronic"] == 0
    assert staged_epoch(opt, s, 2)["lr_feature_phase"] == .01
    assert staged_epoch(opt, s, 11)["lr_electronic"] == pytest.approx(.001)
    end = staged_epoch(opt, s, 100)
    assert end["lr_feature_phase"] == pytest.approx(.0002)
    assert end["hard_balance_weight"] == pytest.approx(.1)
    assert staged_epoch(opt, s, 100) == end


def test_warmstart_reencodes_alpha_instead_of_reinterpreting_old_logit(tmp_path):
    class Fusion(_ScaleMatchedFusionMixin, torch.nn.Module):
        def __init__(self, settings):
            super().__init__()
            self.block1_optical_fusion_logit = torch.nn.Parameter(torch.zeros(()))
            self.block2_optical_fusion_logit = torch.nn.Parameter(torch.zeros(()))
            self._configure_balanced_fusion(settings)

    def model(settings):
        core = torch.nn.Module()
        core.hybrid = Fusion(settings)
        return SimpleNamespace(core=core, head=torch.nn.Linear(1, 1), checkpoint_architecture=architecture_label(settings))

    old = load_settings(TASK / "configs/moe_dc20_mean_only_continue.yaml")
    source = model(old)
    path = tmp_path / "source.pt"
    torch.save({"architecture": source.checkpoint_architecture, "epoch": 75,
                "core": source.core.state_dict(), "saliency_head": source.head.state_dict()}, path)
    target_settings = load_settings(TASK / "configs/moe_staged_alpha_ge040.yaml")
    target_settings.initialization_checkpoint = path
    target = model(target_settings)
    initialize_student(target, target_settings)
    assert float(target.core.hybrid.block1_optical_fusion) == pytest.approx(.45)
    assert float(target.core.hybrid.block2_optical_fusion) == pytest.approx(.45)
    target_settings.reset_fusion_on_warmstart = False
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        initialize_student(target, target_settings)
