from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    FairElectronicAmplitudeRouter,
    sparsify_probabilities,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    DatasetBundle,
    PoseRecord,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.modeling import (
    RobustDenseVision2Core,
    _new_router,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import (
    build_periodic_test_protocol,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.training import (
    ModelEMA,
    should_evaluate_periodic_test,
)


ROOT = Path(__file__).resolve().parents[1]


def _record(source: str, index: int, split: str) -> PoseRecord:
    return PoseRecord(
        sample_id=f"{source}_{index:05d}",
        source=source,
        split=split,
        source_index=index,
        image_path=Path(f"/{source}/{index:05d}.jpg"),
        keypoints=np.zeros((14, 2), dtype=np.float32),
        raw_visibility=np.ones(14, dtype=np.float32),
    )


def test_periodic_test_protocol_keeps_full_canonical_training_split() -> None:
    official = DatasetBundle(
        train=[
            *[_record("lspet", index, "train") for index in range(9428)],
            *[_record("lsp", index, "train") for index in range(1000)],
        ],
        test=[_record("lsp", index + 1000, "test") for index in range(1000)],
        metadata={"protocol": "synthetic"},
    )
    result = build_periodic_test_protocol(official)
    assert (len(result.train), len(result.test)) == (10428, 1000)
    assert sum(record.source == "lspet" for record in result.train) == 9428
    assert sum(record.source == "lsp" for record in result.train) == 1000
    assert result.metadata["validation_samples"] == 0
    assert result.metadata["test_visible_during_training"] is True
    assert result.metadata["test_used_for_checkpoint_selection"] is True
    assert not ({record.sample_id for record in result.train} & {record.sample_id for record in result.test})


def test_periodic_test_schedule_is_epoch1_every5_and_final() -> None:
    observed = [
        epoch
        for epoch in range(1, 99)
        if should_evaluate_periodic_test(epoch, total_epochs=98, interval=5)
    ]
    assert observed == [1, *range(5, 96, 5), 98]


def test_ema_context_is_saved_as_ema_and_restores_live_weights(tmp_path: Path) -> None:
    core = torch.nn.Linear(1, 1, bias=False)
    head = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        core.weight.fill_(1.0)
        head.weight.fill_(2.0)
    ema = ModelEMA(core, head, decay=0.5)
    with torch.no_grad():
        core.weight.fill_(5.0)
        head.weight.fill_(6.0)
    ema.update()

    checkpoint = tmp_path / "ema_state.pt"
    with ema.applied():
        torch.testing.assert_close(core.weight, torch.full_like(core.weight, 3.0))
        torch.testing.assert_close(head.weight, torch.full_like(head.weight, 4.0))
        torch.save({"core": core.state_dict(), "head": head.state_dict()}, checkpoint)

    # Exiting the context restores the live optimizer weights.
    torch.testing.assert_close(core.weight, torch.full_like(core.weight, 5.0))
    torch.testing.assert_close(head.weight, torch.full_like(head.weight, 6.0))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.testing.assert_close(saved["core"]["weight"], torch.full((1, 1), 3.0))
    torch.testing.assert_close(saved["head"]["weight"], torch.full((1, 1), 4.0))


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_power_l2_forward_is_sparse_and_equal_power(top_k: int) -> None:
    probabilities = torch.softmax(torch.randn(7, 4), dim=-1)
    weights, selected, indices = sparsify_probabilities(
        probabilities,
        top_k,
        normalization="power_l2",
        straight_through=True,
    )
    assert tuple(indices.shape) == (7, top_k)
    assert torch.equal(selected.sum(dim=-1), torch.full((7,), top_k))
    assert torch.all(weights.masked_select(~selected) == 0)
    torch.testing.assert_close(
        weights.detach().square().sum(dim=-1), torch.ones(7)
    )


def test_correct_ste_gives_top1_router_gradient() -> None:
    logits = torch.randn(5, 4, requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)
    weights, _, _ = sparsify_probabilities(
        probabilities,
        1,
        normalization="power_l2",
        straight_through=True,
    )
    (weights * torch.arange(1, 5, dtype=weights.dtype)).sum().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_release_configs_share_exact_robust_contract() -> None:
    repository_root = ROOT.parents[1]
    rows = []
    for name in (
        "electronic_power_topk1.yaml",
        "electronic_power_topk2.yaml",
        "electronic_power_topk4.yaml",
        "optical_power_topk2.yaml",
    ):
        settings = load_settings(ROOT / "configs" / "release" / name)
        rows.append(settings)
        assert settings.canvas_size == 518
        assert settings.active_size == 478
        assert settings.expert_size == 224
        assert settings.num_experts == 4
        assert settings.electronic_width == 192
        assert settings.electronic_vision_token_mixer_type == "depthwise_conv2d"
        assert settings.pixel_pitch_um == 17.0
        assert settings.global_to_detector_distance_m == 0.10
        assert settings.optical_fusion_minimum == 0.05
        assert settings.language_optical_max_shift_pixels == 16
        assert settings.router_weight_normalization == "power_l2"
        assert settings.router_straight_through is True
        assert settings.evaluate_test_each_epoch is False
        assert settings.periodic_test_interval_epochs == 5
        assert settings.periodic_test_at_epoch_one is True
        assert settings.periodic_test_at_final_epoch is True
        assert settings.output_dir.name.endswith("_periodic_test5")
        assert settings.data_root == repository_root / "data" / "lsp_pose"
        assert settings.cache_dir == repository_root / "cache" / "qwen"
        assert settings.output_dir.parent == ROOT / "runs"
        assert settings.common_initialization_checkpoint == (
            ROOT / "runs" / "shared_untrained_initialization.pt"
        )
    assert [settings.top_k for settings in rows] == [1, 2, 4, 2]
    assert [settings.router_backend for settings in rows] == [
        "electronic",
        "electronic",
        "electronic",
        "optical",
    ]


def test_modeling_imports_latest_robust_core() -> None:
    assert (
        RobustDenseVision2Core.__init__.__globals__["RobustVisionTwoBlockOpticalCore"].__module__
        == "experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks"
    )


@pytest.mark.parametrize(
    ("config_name", "selected_experts"),
    [
        ("electronic_power_topk1.yaml", 1),
        ("optical_power_topk2.yaml", 2),
    ],
)
def test_robust_core_forward_backward_reaches_router_and_feature_phase(
    config_name: str,
    selected_experts: int,
) -> None:
    """Exercise the real 518/478 robust path, not only Router algebra."""

    torch.manual_seed(7)
    settings = load_settings(ROOT / "configs" / "release" / config_name)
    core = RobustDenseVision2Core(1024, settings)
    core.optical_branch.core.router = _new_router(
        settings, core.optical_branch.core.geometry
    )
    core.train()
    _, latent = core.forward_groups(
        [torch.randn(16, 1024)],
        [(1, 4, 4)],
    )
    assert tuple(latent.shape) == (1, 16, 192)
    assert bool(torch.isfinite(latent).all())
    assert int(core.last_routing["selected_mask"].sum()) == selected_experts

    latent.square().mean().backward()
    router_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in core.router.parameters()
        if parameter.grad is not None
    )
    feature_phase_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in core.expert_layers.parameters()
        if parameter.grad is not None
    )
    assert router_gradient > 0.0
    assert feature_phase_gradient > 0.0
