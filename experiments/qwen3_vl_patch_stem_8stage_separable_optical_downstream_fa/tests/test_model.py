from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    STEM_FORMAT,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    grid_to_qwen_tokens,
)

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.model import (
    P11DownstreamModel,
    ProgressiveDenseHead,
)


def _fake_stem(path: Path) -> Path:
    generator = torch.Generator().manual_seed(31)
    torch.save(
        {
            "format": STEM_FORMAT,
            "conv2d_weight": torch.randn(1024, 3, 16, 16, generator=generator) * 0.001,
            "conv2d_bias": torch.zeros(1024),
            "position_embedding": torch.randn(196, 1024, generator=generator) * 0.01,
            "metadata": {
                "image_size": 224,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            },
        },
        path,
    )
    return path


def _p11_config() -> dict[str, object]:
    return {
        "canvas_size": 224,
        "optical_channels": 3,
        "num_stages": 8,
        "token_dim": 224,
        "num_classes": 1000,
        "head_hidden_dim": 448,
        "phase_init_std": 0.10,
        "optical_gate_init": 0.60,
        "optical_gate_min": 0.50,
        "mixer_width": 96,
        "mixer_expansion": 2.0,
        "mixer_kernel_size": 3,
        "mixer_dropout": 0.10,
        "token_axis_propagation_distance_m": 0.05,
        "channel_axis_propagation_distance_m": 0.05,
        "seed": 2026,
    }


@pytest.fixture()
def source_files(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    stem = _fake_stem(tmp_path / "stem.pt")
    torch.manual_seed(19)
    source = QwenStemSeparableOpticalImageNetBackbone(stem, _p11_config())
    checkpoint = tmp_path / "p11_backbone.pt"
    torch.save(
        {
            "backbone": source.backbone_state_dict(),
            "best_epoch": 88,
            "stem_checkpoint_sha256": source.stem.checkpoint_sha256,
            "model_report": source.parameter_report(),
        },
        checkpoint,
    )
    return stem, checkpoint, _p11_config()


def _build(
    source_files: tuple[Path, Path, dict[str, object]],
    task: str = "caltech101",
) -> P11DownstreamModel:
    stem, checkpoint, config = source_files
    return P11DownstreamModel(
        stem_checkpoint=stem,
        source_checkpoint=checkpoint,
        p11_config=config,
        task=task,  # type: ignore[arg-type]
    )


def test_strict_p11_load_removes_old_readout_and_reports_parameters(
    source_files: tuple[Path, Path, dict[str, object]],
) -> None:
    model = _build(source_files)
    assert isinstance(model.backbone.readout, nn.Identity)
    assert list(model.backbone.readout.parameters()) == []
    assert torch.equal(
        model.backbone.p11_separable_architecture_signature.cpu(),
        torch.tensor([11, 1, 2, 4]),
    )
    assert not any(parameter.requires_grad for parameter in model.backbone.stem.parameters())
    phase_ids = {id(parameter) for parameter in model.phase_parameters()}
    electronic_ids = {id(parameter) for parameter in model.backbone_parameters()}
    assert phase_ids
    assert electronic_ids
    assert phase_ids.isdisjoint(electronic_ids)

    report = model.parameter_report()
    assert report["optical_phase_parameters"] == 8 * 3 * 224 * 224
    assert report["optical_fraction_of_reusable_backbone"] >= 0.50
    assert report["old_imagenet_readout_parameters"] == 0
    assert report["global_descriptor_dim"] == 448
    assert report["retrieval_embedding_dim"] == 256


def test_wrong_signature_and_partial_backbone_are_rejected(
    source_files: tuple[Path, Path, dict[str, object]], tmp_path: Path
) -> None:
    stem, checkpoint, config = source_files
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    wrong_signature = dict(payload)
    wrong_signature["backbone"] = dict(payload["backbone"])
    wrong_signature["backbone"]["p11_separable_architecture_signature"] = torch.tensor(
        [9, 0, 0, 0]
    )
    wrong_path = tmp_path / "wrong_signature.pt"
    torch.save(wrong_signature, wrong_path)
    with pytest.raises(RuntimeError, match="P11 architecture signature"):
        P11DownstreamModel(
            stem_checkpoint=stem,
            source_checkpoint=wrong_path,
            p11_config=config,
            task="caltech101",
        )

    partial = dict(payload)
    partial["backbone"] = dict(payload["backbone"])
    adapter_key = next(name for name in partial["backbone"] if name.startswith("adapter."))
    partial["backbone"].pop(adapter_key)
    partial_path = tmp_path / "partial.pt"
    torch.save(partial, partial_path)
    with pytest.raises(RuntimeError, match="load was incomplete"):
        P11DownstreamModel(
            stem_checkpoint=stem,
            source_checkpoint=partial_path,
            p11_config=config,
            task="caltech101",
        )


@pytest.mark.parametrize(
    ("task", "expected_shape", "expected_keys"),
    [
        ("caltech101", (2, 101), {"logits", "embedding"}),
        ("isic2016", (2, 1, 224, 224), {"logits"}),
        ("lsp", (2, 14, 56, 56), {"logits"}),
    ],
)
def test_unified_forward_outputs_and_temporary_head_budget(
    source_files: tuple[Path, Path, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    expected_shape: tuple[int, ...],
    expected_keys: set[str],
) -> None:
    model = _build(source_files, task)
    final = torch.randn(2, 3, 224, 224)
    monkeypatch.setattr(
        model,
        "forward_features",
        lambda images, ablation="normal": (final[: images.shape[0]], ()),
    )
    output = model(torch.zeros(2, 3, 224, 224))
    assert set(output) == expected_keys
    assert tuple(output["logits"].shape) == expected_shape
    if task == "caltech101":
        assert tuple(output["embedding"].shape) == (2, 256)
        torch.testing.assert_close(
            output["embedding"].norm(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-5
        )
    else:
        assert sum(parameter.numel() for parameter in model.head_parameters()) < 1_000_000
        assert model.parameter_report()["dense_head_below_one_million"] is True


def test_dense_head_restores_qwen_tokens_to_true_spatial_grid() -> None:
    head = ProgressiveDenseHead(
        token_count=196,
        token_dim=224,
        optical_banks=3,
        output_channels=1,
        output_size=56,
    )
    grid = torch.arange(196, dtype=torch.float32).view(1, 1, 14, 14)
    qwen = grid_to_qwen_tokens(grid).expand(1, 196, 224)
    field = torch.zeros(1, 3, 224, 224)
    field[:, :, :196] = qwen.unsqueeze(1).expand(-1, 3, -1, -1)
    restored = head.fused_grid(field)
    torch.testing.assert_close(restored[:, :1], grid)


def test_feedback_random_is_reproducible_and_resume_reconstructs_it(
    source_files: tuple[Path, Path, dict[str, object]],
) -> None:
    model = _build(source_files)
    first = model.configure_feedback("fa_random", random_seed=77)
    second = model.configure_feedback("fa_random", random_seed=77)
    different = model.configure_feedback("fa_random", random_seed=78)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert not any(name.endswith("feedback_phase") for name in model.state_dict())

    resumed = _build(source_files)
    resumed.load_state_dict(model.state_dict(), strict=True)
    reconstructed = resumed.configure_feedback("fa_random", random_seed=77)
    assert torch.equal(reconstructed, first)
    pretrained = resumed.configure_feedback("fa_pretrained")
    assert torch.equal(pretrained, resumed.source_phases.cpu())
    # Token-stage phases already live in physical row-major layout and must not
    # be permuted by the downstream feedback configurator.
    assert torch.equal(
        resumed.backbone.stages[0].feedback_phase.cpu(), resumed.source_phases[0].cpu()
    )


def _stage_chain_gradients(
    model: P11DownstreamModel,
    method: str,
    amplitude: torch.Tensor,
    probe: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    model.eval()
    model.configure_feedback(method, random_seed=123)  # type: ignore[arg-type]
    model.zero_grad(set_to_none=True)
    value = amplitude.detach().clone().requires_grad_(True)
    output = value
    for stage in model.backbone.stages:
        output = stage(output)
    (output * probe).mean().backward()
    gradients = [stage.raw_phase.grad.detach().clone() for stage in model.backbone.stages]
    assert value.grad is not None
    return output.detach(), gradients, value.grad.detach().clone()


def test_pretrained_feedback_matches_bp_at_source_and_random_last_stage_is_exact(
    source_files: tuple[Path, Path, dict[str, object]],
) -> None:
    torch.manual_seed(41)
    model = _build(source_files)
    amplitude = torch.rand(1, 3, 224, 224)
    probe = torch.randn_like(amplitude)

    bp_output, bp_gradients, bp_input_gradient = _stage_chain_gradients(
        model, "bp_current", amplitude, probe
    )
    pretrained_output, pretrained_gradients, pretrained_input_gradient = (
        _stage_chain_gradients(model, "fa_pretrained", amplitude, probe)
    )
    torch.testing.assert_close(bp_output, pretrained_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        bp_input_gradient, pretrained_input_gradient, rtol=5e-4, atol=5e-5
    )
    for exact, fixed in zip(bp_gradients, pretrained_gradients, strict=True):
        assert float(F.cosine_similarity(exact.flatten(), fixed.flatten(), dim=0)) > 0.9999
        torch.testing.assert_close(exact, fixed, rtol=5e-4, atol=5e-5)

    random_output, random_gradients, _ = _stage_chain_gradients(
        model, "fa_random", amplitude, probe
    )
    torch.testing.assert_close(bp_output, random_output, rtol=2e-5, atol=2e-6)
    # Stage eight receives the task loss directly, so its local phase update is
    # exact even for random feedback. Only earlier stages cross a random
    # connector.
    torch.testing.assert_close(
        bp_gradients[-1], random_gradients[-1], rtol=5e-4, atol=5e-5
    )
    earlier_cosines = [
        float(F.cosine_similarity(exact.flatten(), random.flatten(), dim=0))
        for exact, random in zip(bp_gradients[:-1], random_gradients[:-1], strict=True)
    ]
    assert min(earlier_cosines) < 0.99
