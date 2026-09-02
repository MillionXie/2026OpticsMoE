from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.cache_qwen_inputs import (
    qwen_pool_premerger_tokens,
    qwen_patch_with_position,
)
from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.modeling import (
    ScaleMatchedFusion,
)
from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.prepare_manifest import (
    _flatten_quality_records,
)
from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.run import (
    synthetic_smoke,
)
from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.settings import (
    load_settings,
)
from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.training import (
    _optimizer,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_contracts() -> None:
    for name, backend, top_k in (
        ("e1.yaml", "electronic", 1),
        ("e2.yaml", "electronic", 2),
        ("e4.yaml", "electronic", 4),
        ("o2.yaml", "optical", 2),
    ):
        settings = load_settings(ROOT / "configs/release" / name)
        assert settings.router_backend == backend
        assert settings.top_k == top_k
        assert settings.router_weight_normalization == "power_l2"
        assert settings.router_straight_through
        assert settings.router_detector_intervals == ((164, 223), (255, 314))
        assert settings.target_names == ("spatial", "temporal")
        assert "alignment" not in settings.prompt.lower()


def test_balanced_post_rescale_matches_electronic_rms() -> None:
    settings = load_settings(ROOT / "configs/release/e2.yaml", synthetic=True)
    fusion = ScaleMatchedFusion(settings)
    electronic = torch.randn(3, 4, 5, 8)
    optical = torch.randn_like(electronic) * 100.0
    mask = torch.ones(3, 4, 5, dtype=torch.bool)
    fused = fusion(electronic, optical, mask)
    torch.testing.assert_close(
        fused.square().mean((1, 2, 3)).sqrt(),
        electronic.square().mean((1, 2, 3)).sqrt(),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert 0.01 < float(fusion.alpha) < 0.49


def test_position_embedding_uses_supported_fast_api() -> None:
    class Visual:
        def patch_embed(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def fast_pos_embed_interpolate(self, _grid: torch.Tensor) -> torch.Tensor:
            return torch.ones(2, 3)

    output = qwen_patch_with_position(Visual(), torch.zeros(2, 3), torch.ones(1, 3))
    assert torch.equal(output, torch.ones(2, 3))
    with pytest.raises(RuntimeError, match="fast_pos_embed_interpolate"):
        qwen_patch_with_position(SimpleNamespace(patch_embed=lambda value: value), torch.zeros(2, 3), torch.ones(1, 3))


def test_qwen_block_major_premerger_pooling() -> None:
    hidden = torch.arange(2 * 784 * 1024, dtype=torch.float32).reshape(-1, 1024)
    pooled = qwen_pool_premerger_tokens(hidden, image_count=2)
    assert pooled.shape == (2, 196, 1024)
    torch.testing.assert_close(pooled[0, 0], hidden[:4].mean(0))
    torch.testing.assert_close(pooled[1, -1], hidden[-4:].mean(0))


def test_nested_lgvq_prompt_schema_is_flattened_and_deduplicated() -> None:
    row = {
        "path": "videos/example.mp4",
        "prompt": "example prompt",
        "spatial_quality": 61.0,
        "temporal_quality": 58.0,
    }
    records = _flatten_quality_records({"part_a": [row], "part_b": {"copy": [row]}})
    assert records == [row]


def test_synthetic_e2_o2_smoke_and_optimizer_partition() -> None:
    settings = load_settings(ROOT / "configs/release/e2.yaml", synthetic=True)
    report = synthetic_smoke(settings)
    assert report["status"] == "passed"
    assert set(report["variants"]) == {"electronic", "optical"}
    assert not report["alignment_output"]
    electronic_groups = report["variants"]["electronic"]["optimizer_groups"]
    optical_groups = report["variants"]["optical"]["optimizer_groups"]
    assert electronic_groups["router"]["lr"] == pytest.approx(0.001)
    assert "optical_router_phase" not in electronic_groups
    assert optical_groups["phase"]["lr"] == pytest.approx(0.006)
    assert optical_groups["optical_router_phase"]["lr"] == pytest.approx(0.01)
    assert "router" not in optical_groups
    assert optical_groups["optical_router_phase"]["parameters"] > 0
