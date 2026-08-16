from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicSequenceCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.settings import (
    load_settings,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        electronic_width=32,
        electronic_heads=4,
        electronic_ff_multiplier=2.0,
        electronic_dropout=0.0,
        electronic_attention_dropout=0.0,
        electronic_layers=2,
        electronic_initial_residual_weight=0.1,
    )


def test_dense_electronic_core_preserves_packed_shape_and_has_no_moe() -> None:
    core = ElectronicSequenceCore(48, 8, _settings())
    groups = [torch.randn(3, 48), torch.randn(5, 48)]
    packed, latent = core.forward_groups(groups, causal=True)
    assert packed.shape == (8, 48)
    assert latent.shape == (2, 5, 32)
    assert list(core.router.parameters()) == []
    breakdown = core.parameter_breakdown()
    assert breakdown["moe_enabled"] is False
    assert breakdown["optical_parameters"] == 0
    assert breakdown["router_parameters"] == 0
    assert not any("raw_phase" in name for name, _ in core.named_parameters())


def test_electronic_readout_returns_unit_64d_embeddings() -> None:
    readout = ElectronicRetrievalReadout(32, 64, 48, 0.0)
    output = readout(torch.randn(4, 32))
    assert output.shape == (4, 64)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1.0e-5)


def test_release_config_is_direct_target10_without_optical_or_moe() -> None:
    settings = load_settings(
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/"
        "configs/release/caltech101_target10_electronic.yaml"
    )
    assert len(settings.selected_skus) == 10
    assert settings.use_all_categories is False
    assert settings.optimizer_steps_per_epoch is None
    assert settings.num_experts == settings.top_k == 1
    assert settings.lambda_router_balance == 0.0
    assert settings.lambda_router_importance == 0.0
    assert settings.phase_learning_rate == 0.0
    assert settings.learning_rate == 1.5e-4
    assert settings.weight_decay == 0.01
