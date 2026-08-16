from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicSequenceCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    _learning_rate_scale,
    episodic_prototype_retrieval_loss,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.settings import (
    load_settings,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        electronic_width=32,
        electronic_expansion=2.0,
        electronic_dropout=0.0,
        electronic_layers=2,
        electronic_initial_residual_weight=0.1,
        electronic_token_mixer_enabled=False,
        electronic_token_mixer_kernel_size=5,
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
    assert breakdown["attention_enabled"] is False
    assert breakdown["attention_parameters"] == 0
    assert breakdown["optical_parameters"] == 0
    assert breakdown["router_parameters"] == 0
    assert not any("raw_phase" in name for name, _ in core.named_parameters())


def test_electronic_readout_returns_unit_64d_embeddings() -> None:
    readout = ElectronicRetrievalReadout(32, 64)
    output = readout(torch.randn(4, 32))
    assert output.shape == (4, 64)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1.0e-5)


def test_token_mixer_uses_tokens_and_masks_padding() -> None:
    settings = _settings()
    settings.electronic_token_mixer_enabled = True
    core = ElectronicSequenceCore(48, 8, settings)
    groups = [torch.randn(3, 48), torch.randn(5, 48)]
    packed, latent = core.forward_groups(groups, causal=True)
    assert packed.shape == (8, 48)
    assert latent.shape == (2, 5, 32)
    assert torch.count_nonzero(latent[0, 3:]) == 0
    breakdown = core.parameter_breakdown()
    assert breakdown["token_mixing_enabled"] is True
    assert breakdown["token_mixer_parameters"] > 0


def test_v2_config_enables_mean_max_token_mixing_without_teacher() -> None:
    settings = load_settings(
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/"
        "configs/release/caltech101_target10_electronic_token_mixer.yaml"
    )
    assert settings.teacher_enabled is False
    assert settings.electronic_width == 192
    assert settings.electronic_layers == 3
    assert settings.electronic_token_mixer_enabled is True
    assert settings.electronic_token_mixer_kernel_size == 5
    assert settings.electronic_pooling == "mean_max"
    assert settings.detector_output_size == 384
    assert settings.lambda_kd == 0.0
    assert settings.lambda_relational_kd == 0.0
    assert settings.lambda_teacher_gallery == 0.0


def test_release_config_is_direct_target10_without_optical_or_moe() -> None:
    settings = load_settings(
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/"
        "configs/release/caltech101_target10_electronic.yaml"
    )
    assert len(settings.selected_skus) == 10
    assert settings.use_all_categories is False
    assert settings.optimizer_steps_per_epoch is None
    assert settings.train_limit_per_sku is None
    assert settings.test_limit_per_sku == 20
    assert settings.reserve_test_before_train is True
    assert settings.num_experts == settings.top_k == 1
    assert settings.lambda_router_balance == 0.0
    assert settings.lambda_router_importance == 0.0
    assert settings.phase_learning_rate == 0.0
    assert settings.lambda_kd == 0.0
    assert settings.lambda_relational_kd == 0.0
    assert settings.lambda_teacher_gallery == 0.0
    assert settings.lambda_ret == 1.0
    assert settings.lambda_gallery == 1.0
    assert settings.episodic_prototype_loss_enabled is True
    assert settings.electronic_width == 128
    assert settings.electronic_layers == 2
    assert settings.learning_rate == 1.5e-4
    assert settings.weight_decay == 0.01


def test_episodic_prototype_loss_updates_supports_and_queries() -> None:
    embeddings = torch.randn(30, 64, requires_grad=True)
    labels = torch.arange(10).repeat_interleave(3)
    loss, logits, targets = episodic_prototype_retrieval_loss(
        embeddings, labels, 0.15
    )
    assert logits.shape == (20, 10)
    assert targets.shape == (20,)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.all(embeddings.grad.norm(dim=-1) > 0)


def test_cosine_schedule_warms_up_then_decays() -> None:
    settings = SimpleNamespace(
        learning_rate_schedule="cosine", learning_rate_warmup_ratio=0.1
    )
    assert _learning_rate_scale(settings, 0, 100) == 0.1
    assert _learning_rate_scale(settings, 9, 100) == 1.0
    assert _learning_rate_scale(settings, 10, 100) == 1.0
    assert _learning_rate_scale(settings, 100, 100) == 0.0


def test_teacher_ablation_changes_only_teacher_kd_and_output() -> None:
    no_teacher = load_settings(
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/"
        "configs/release/caltech101_target10_electronic.yaml"
    )
    teacher = load_settings(
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/"
        "configs/release/caltech101_target10_mlp_teacher_kd.yaml"
    )
    assert teacher.teacher_enabled is True
    assert teacher.lambda_kd == 1.0
    assert teacher.lambda_relational_kd == 0.0
    assert teacher.lambda_teacher_gallery == 0.0
    for name in (
        "selected_skus",
        "train_limit_per_sku",
        "test_limit_per_sku",
        "epochs",
        "learning_rate",
        "adapter_learning_rate",
        "readout_learning_rate",
        "electronic_width",
        "electronic_layers",
        "electronic_expansion",
        "lambda_ret",
        "lambda_gallery",
    ):
        assert getattr(teacher, name) == getattr(no_teacher, name)
    assert teacher.output_dir != no_teacher.output_dir
    assert teacher.teacher_cache_path != no_teacher.teacher_cache_path
