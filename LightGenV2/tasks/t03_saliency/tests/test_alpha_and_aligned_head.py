from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency.aligned_baseline import AlignedReadout
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.training import staged_epoch
from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import _range_gate

TASK = Path(__file__).resolve().parents[1]


def test_aligned_head_counts_and_forward():
    head = AlignedReadout()
    assert head.parameter_audit() == {"adapter": 197184, "decoder": 85412, "total": 282596}
    with torch.no_grad():
        assert head(torch.randn(1, 1024, 14, 14)).shape == (1, 1, 224, 224)


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
