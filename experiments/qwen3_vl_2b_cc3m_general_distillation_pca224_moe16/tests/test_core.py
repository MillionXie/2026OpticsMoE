from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16.data_prepare import (
    extract_webdataset_shard,
)
from experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16.datasets import (
    read_manifest,
)
from experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16.optics.moe import (
    LanguagePCAOpticalMoE,
    PCAHomogeneousMoEOpticalCore,
    VisionPCAOpticalMoE,
)
from experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16.pca import (
    FixedPCAProjection,
    StreamingPCAFitter,
    load_projection,
    save_projection,
)
from experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16.teacher_cache import (
    collate_cached_rows,
)
from experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16.training import (
    compute_stage_losses,
    masked_tokenwise_normalized_mse,
)


def _projection(input_dim: int, latent_dim: int = 8) -> FixedPCAProjection:
    generator = torch.Generator().manual_seed(input_dim)
    matrix = torch.randn(input_dim, latent_dim, generator=generator)
    components = torch.linalg.qr(matrix, mode="reduced").Q
    return FixedPCAProjection(torch.zeros(input_dim), components)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        latent_dim=8,
        max_visual_tokens=8,
        max_language_tokens=8,
        canvas_size=22,
        active_size=18,
        expert_size=8,
        expert_pitch=10,
        num_experts=4,
        expert_grid_rows=2,
        expert_grid_cols=2,
        expert_layers=4,
        top_k=2,
        router_pool_size=2,
        router_temperature=1.0,
        router_input_layernorm_enabled=True,
        router_input_layernorm_eps=1e-5,
        amplitude_slm_weight_domain="amplitude",
        amplitude_slm_input_normalization="none",
        wavelength_nm=532.0,
        pixel_pitch_um=8.0,
        expert_interlayer_distance_m=0.01,
        last_expert_to_global_distance_m=0.01,
        global_to_detector_distance_m=0.01,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        phase_init_std=0.02,
        phase_dropout_mode="none",
        phase_dropout_p=0.0,
        phase_dropout_block_size=2,
        phase_dropout_batch_shared=True,
        k_space_constraint_enabled=False,
        theta_max_deg=1.0,
        detector_output_size=8,
        detector_layernorm_eps=1e-5,
        detector_layernorm_affine=False,
        detector_layernorm_scope="per_token",
    )


def test_pca_buffers_have_no_gradient_and_encode_decode_shapes() -> None:
    projection = _projection(12)
    hidden = torch.randn(3, 5, 12, requires_grad=True)
    latent = projection.encode(hidden)
    reconstructed = projection.decode(latent)
    assert latent.shape == (3, 5, 8)
    assert reconstructed.shape == hidden.shape
    assert list(projection.parameters()) == []
    assert projection.mean.requires_grad is False
    assert projection.components.requires_grad is False
    reconstructed.square().mean().backward()
    assert hidden.grad is not None
    assert projection.mean.grad is None
    assert projection.components.grad is None


def test_each_stack_uses_one_shared_pca_object() -> None:
    settings = _settings()
    vision_pca = _projection(12)
    language_pca = _projection(16)
    vision = VisionPCAOpticalMoE(vision_pca, settings)
    language = LanguagePCAOpticalMoE(language_pca, settings)
    assert vision.pca is vision_pca
    assert language.pca is language_pca
    assert not any("stage_pca" in name for name, _ in vision.named_modules())
    assert not any("stage_pca" in name for name, _ in language.named_modules())


def test_no_trainable_hidden_latent_linear_adapters() -> None:
    settings = _settings()
    modules = [
        VisionPCAOpticalMoE(_projection(12), settings),
        LanguagePCAOpticalMoE(_projection(16), settings),
    ]
    for module, hidden_dim in zip(modules, (12, 16)):
        forbidden = []
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Linear) and {
                child.in_features,
                child.out_features,
            } == {hidden_dim, settings.latent_dim}:
                forbidden.append(name)
        assert forbidden == []
        report = module.parameter_breakdown()
        assert report["trainable_hidden_to_latent_linear_parameters"] == 0
        assert report["trainable_latent_to_hidden_linear_parameters"] == 0


def test_signed_readout_is_loss_feature_and_reload_is_nonnegative() -> None:
    settings = _settings()
    core = PCAHomogeneousMoEOpticalCore(settings.max_visual_tokens, settings)
    groups = [torch.randn(6, settings.latent_dim)]
    fields = core.encode_groups(groups)
    state = core.start(fields)
    signed = core.run_stage(0, state)
    assert signed.shape == (1, 8, 8)
    assert torch.any(signed < 0), "LayerNorm signed readout should retain negative values"
    assert state.reload_amplitude is not None
    assert torch.all(state.reload_amplitude >= 0)
    assert torch.equal(state.reload_amplitude, F.relu(signed))
    # The stage feature is the signed tensor itself, not its ReLU reload.
    packed = core.pack_fields(signed, [6])
    assert torch.any(packed < 0)


def test_zero_padding_and_token_overflow() -> None:
    core = PCAHomogeneousMoEOpticalCore(8, _settings())
    fields = core.encode_groups([torch.randn(6, 8)])
    assert fields.shape == (1, 8, 8)
    assert torch.count_nonzero(fields[:, 6:]) == 0
    with pytest.raises(RuntimeError, match="token count 9 exceeds optical field rows=8"):
        core.encode_groups([torch.randn(9, 8)])


def test_language_sequence_overflow_is_explicit() -> None:
    language = LanguagePCAOpticalMoE(_projection(16), _settings())
    with pytest.raises(RuntimeError, match="language sequence length 9 exceeds"):
        language.set_attention_mask(torch.ones(1, 9))


def test_padding_tokens_do_not_affect_masked_loss() -> None:
    teacher = torch.randn(2, 5, 8)
    student = teacher.clone()
    student[:, 3:] = 1000.0
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    loss = masked_tokenwise_normalized_mse(student, teacher, mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


def test_vision_language_and_joint_minimal_forward_backward() -> None:
    settings = _settings()
    vision = VisionPCAOpticalMoE(_projection(12), settings)
    language = LanguagePCAOpticalMoE(_projection(16), settings)
    vision_hidden = torch.randn(6, 12)
    vision.compute(vision_hidden, torch.tensor([0, 6], dtype=torch.int32))
    assert len(vision.stage_latents) == 4
    assert all(tap.shape == (6, 8) for tap in vision.stage_latents)

    language.set_attention_mask(torch.ones(1, 5, dtype=torch.long))
    hidden = torch.randn(1, 5, 16)
    for stage in range(4):
        hidden = language.forward_stage(stage, hidden)
        if stage < 3:
            hidden = hidden + 0.01
    assert len(language.stage_latents) == 4
    assert all(tap.shape == (5, 8) for tap in language.stage_latents)

    vision_loss = sum(tap.square().mean() for tap in vision.stage_latents)
    language_loss = sum(tap.square().mean() for tap in language.stage_latents)
    vision_loss.backward(retain_graph=True)
    assert any(parameter.grad is not None for parameter in vision.parameters() if parameter.requires_grad)
    language_loss.backward()
    assert any(parameter.grad is not None for parameter in language.parameters() if parameter.requires_grad)

    vision.zero_grad(set_to_none=True)
    language.zero_grad(set_to_none=True)
    vision.compute(vision_hidden, torch.tensor([0, 6], dtype=torch.int32))
    language.set_attention_mask(torch.ones(1, 5, dtype=torch.long))
    hidden = torch.randn(1, 5, 16)
    for stage in range(4):
        hidden = language.forward_stage(stage, hidden)
    joint = sum(tap.square().mean() for tap in vision.stage_latents)
    joint = joint + sum(tap.square().mean() for tap in language.stage_latents)
    joint.backward()
    assert any(parameter.grad is not None for parameter in vision.parameters() if parameter.requires_grad)
    assert any(parameter.grad is not None for parameter in language.parameters() if parameter.requires_grad)


def test_separate_vision_loss_does_not_require_language_router_state() -> None:
    settings = _settings()
    vision = VisionPCAOpticalMoE(_projection(12), settings)
    language = LanguagePCAOpticalMoE(_projection(16), settings)
    vision.compute(torch.randn(6, 12), torch.tensor([0, 6], dtype=torch.int32))
    replacement = SimpleNamespace(
        vision_surrogate=vision,
        language_surrogate=language,
    )
    batch = {
        "vision_targets": [tap.detach().clone() for tap in vision.stage_latents],
        "language_targets": [],
        "language_mask": torch.empty(0, dtype=torch.bool),
    }
    losses = compute_stage_losses(replacement, batch, "vision")
    assert float(losses["vision"]) == pytest.approx(0.0, abs=1e-7)
    assert float(losses["language"]) == 0.0


def test_synthetic_100_sample_pca_cache_and_train_step(tmp_path: Path) -> None:
    fitter = StreamingPCAFitter(
        input_dim=12,
        latent_dim=8,
        max_tokens=400,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(42)
    raw_samples = [torch.randn(4, 12, generator=generator) for _ in range(100)]
    for sample in raw_samples:
        fitter.update(sample)
    projection, report = fitter.finalize()
    path = tmp_path / "vision.pt"
    save_projection(
        path,
        projection,
        report,
        stack="vision",
        seed=42,
        source_taps=["input", 1, 2, 3, 4],
    )
    loaded = load_projection(path, 12)
    assert loaded.encode(raw_samples[0]).shape == (4, 8)

    rows = []
    for index in range(100):
        targets = [loaded.encode(raw_samples[index]) for _ in range(4)]
        rows.append({
            "sample_indices": torch.tensor(index),
            "input_ids": torch.tensor([1, 2, 3]),
            "pixel_values": torch.randn(2, 3),
            "image_grid_thw": torch.tensor([1, 1, 2]),
            "visual_token_counts": torch.tensor(4),
            "language_token_masks": torch.ones(3, dtype=torch.bool),
            "teacher_vision_input_pca": targets[0],
            "teacher_vision_stage_taps_pca": targets,
            "teacher_language_input_pca": torch.randn(3, 8),
            "teacher_language_stage_taps_pca": [torch.randn(3, 8) for _ in range(4)],
        })
    cache_path = tmp_path / "synthetic_cache.pt"
    torch.save({"rows": rows}, cache_path)
    restored = torch.load(cache_path, map_location="cpu", weights_only=True)["rows"]
    batch = collate_cached_rows(restored[:2])
    assert batch["vision_targets"][0].shape == (8, 8)
    assert batch["language_targets"][0].shape == (2, 3, 8)

    settings = _settings()
    student = VisionPCAOpticalMoE(loaded, settings)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    student.compute(torch.cat(raw_samples[:2]), torch.tensor([0, 4, 8]))
    loss = sum(
        masked_tokenwise_normalized_mse(tap, target)
        for tap, target in zip(student.stage_latents, batch["vision_targets"])
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert math_is_finite(float(loss))


def test_cc3m_webdataset_shard_extraction_builds_jsonl(tmp_path: Path) -> None:
    archive_path = tmp_path / "cc3m-train-0000.tar"
    image_bytes = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(image_bytes, format="JPEG")
    caption = b"a small synthetic image"
    metadata = json.dumps({"caption": caption.decode(), "status": "success"}).encode()
    with tarfile.open(archive_path, "w") as archive:
        for name, payload in (
            ("000000001.jpg", image_bytes.getvalue()),
            ("000000001.json", metadata),
            ("000000001.txt", caption),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    output_dir = tmp_path / "images" / "train" / archive_path.stem
    shard_manifest = tmp_path / f"{archive_path.stem}.jsonl"
    result = extract_webdataset_shard(
        archive_path,
        output_dir,
        shard_manifest,
        source_split="train",
        manifest_root=tmp_path,
    )
    assert result["samples"] == 1
    rows, digest = read_manifest(shard_manifest)
    assert len(rows) == 1
    assert rows[0].sample_id == "cc3m-train-0000:000000001"
    assert rows[0].caption == caption.decode()
    assert rows[0].image_path.is_file()
    assert len(digest) == 64


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
