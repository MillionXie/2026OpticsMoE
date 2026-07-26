from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 import (
    TASK_PROMPTS,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.cache_paths import (
    cache_identity,
    cache_identity_digest,
    precompute_cache_root,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.data_prepare import (
    ensure_kadid10k_dataset,
    inspect_kadid10k_root,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.datasets import (
    load_kadid10k,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.modeling import (
    build_head,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.optics.geometry import (
    Aperture,
    MoEGeometry,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.optics.moe import (
    FullPlaneReadout,
    HomogeneousMoEOpticalCore,
    LanguageDeepStackHomogeneousMoE,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.optics.physical import (
    AngularSpectrumPropagator,
    PhaseLayer,
    aperture_linear_indices,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.optics.router import (
    ElectronicAmplitudeRouter,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.settings import (
    load_settings,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.teacher_cache import (
    expected_metadata,
)
from experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224.training import (
    _reference_disjoint_head_split,
    norm_in_norm_loss,
    pairwise_ranking_loss,
    score_metrics,
)


ROOT = Path(
    "experiments/"
    "qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224"
)
CONFIGS = ROOT / "configs"


def _settings(tmp_path: Path):
    settings = load_settings(CONFIGS / "kadid10k_mos_smoke.json")
    settings.data_root = tmp_path / "kadid10k"
    settings.output_dir = tmp_path / "run"
    settings.precompute_cache_dir = tmp_path / "cache"
    settings.annotations_file = None
    settings.image_dir = None
    settings.download = False
    settings.require_official_counts = False
    settings.train_image_limit = None
    settings.test_image_limit = None
    return settings


def _write_synthetic_kadid(
    root: Path,
    references: int = 6,
    images_per_reference: int = 5,
) -> None:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = ["dist_img,ref_img,dmos,var"]
    for reference in range(1, references + 1):
        for offset in range(images_per_reference):
            distortion_type = offset // 5 + 1
            level = offset % 5 + 1
            name = f"I{reference:02d}_{distortion_type:02d}_{level:02d}.png"
            Image.new(
                "RGB",
                (12, 10),
                (reference * 10, level * 20, offset),
            ).save(images / name)
            score = 1.0 + 4.0 * offset / max(images_per_reference - 1, 1)
            rows.append(f"{name},I{reference:02d}.png,{score:.6f},0.01")
    (root / "dmos.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_main_and_smoke_configs_are_complete() -> None:
    main = load_settings(CONFIGS / "kadid10k_mos.json")
    smoke = load_settings(CONFIGS / "kadid10k_mos_smoke.json")
    for settings in (main, smoke):
        assert settings.dataset == "kadid10k_mos"
        assert settings.task_name == "DMOS"
        assert settings.classification_prompt == TASK_PROMPTS["DMOS"]
        assert settings.download
        assert settings.download_source == "official_url"
        assert settings.download_url.endswith("/kadid10k.zip")
        assert settings.processor_min_pixels == 37632
        assert settings.processor_max_pixels == 37632
        assert settings.max_visual_tokens == 224
        assert settings.max_language_tokens == 224
        assert settings.student_language_mode == "optical_moe"
        assert not settings.native_pre_attention_enabled
        assert settings.transformer_residual_enabled
        assert settings.loss_hidden_weight > 0
        assert settings.loss_answer_weight > 0
        assert settings.loss_prediction_distill_weight > 0
        assert settings.loss_norm_in_norm_weight == pytest.approx(0.1)
        assert settings.loss_ranking_weight == pytest.approx(0.1)
    assert smoke.epochs == 1
    assert smoke.train_image_limit == 16
    assert smoke.test_image_limit == 8


def test_kadid_csv_aliases_normalization_and_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_synthetic_kadid(settings.data_root)
    bundle = load_kadid10k(settings)
    assert bundle.metadata["image_column"] == "dist_img"
    assert bundle.metadata["reference_column"] == "ref_img"
    assert bundle.metadata["score_column"] == "dmos"
    assert bundle.metadata["variance_column"] == "var"
    assert bundle.metadata["quality_score_higher_is_better"] is True
    assert bundle.metadata["label_scale"] == [1.0, 5.0]
    image, normalized = bundle.train[0]
    assert image.mode == "RGB"
    assert 0.0 <= normalized <= 1.0
    metadata = bundle.train.sample_metadata(0)
    assert metadata["image_name"].endswith(".png")
    assert metadata["reference_image"].endswith(".png")
    assert 1.0 <= metadata["dmos"] <= 5.0
    assert metadata["distortion_level"] in {1, 2, 3, 4, 5}


def test_reference_disjoint_split_is_persistent_and_deterministic(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.train_fraction = 2.0 / 3.0
    _write_synthetic_kadid(settings.data_root, references=6)
    first = load_kadid10k(settings)
    second = load_kadid10k(settings)
    train_references = {record.reference_id for record in first.train_records}
    test_references = {record.reference_id for record in first.test_records}
    assert not train_references & test_references
    assert len(train_references) == 4
    assert len(test_references) == 2
    assert first.metadata["split_digest"] == second.metadata["split_digest"]
    payload = json.loads((settings.output_dir / "data_split.json").read_text())
    assert payload["split_unit"] == "reference_image"
    assert payload["train_reference_count"] == 4
    assert payload["test_reference_count"] == 2


def test_teacher_head_internal_validation_is_reference_disjoint(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _write_synthetic_kadid(settings.data_root, references=10)
    dataset = load_kadid10k(settings).train
    train_indices, validation_indices = _reference_disjoint_head_split(
        dataset,
        fraction=0.2,
        seed=42,
    )
    train_refs = {
        dataset.records[index].reference_id for index in train_indices.tolist()
    }
    validation_refs = {
        dataset.records[index].reference_id
        for index in validation_indices.tolist()
    }
    assert train_refs
    assert validation_refs
    assert not train_refs & validation_refs


def test_missing_required_metadata_columns_reports_available_columns(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.data_root.mkdir(parents=True)
    (settings.data_root / "dmos.csv").write_text(
        "dist_img,dmos\nI01_01_01.png,3.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="reference"):
        load_kadid10k(settings)


def test_official_count_guard_rejects_incomplete_dataset(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.require_official_counts = True
    _write_synthetic_kadid(settings.data_root)
    with pytest.raises(RuntimeError, match="official-count validation"):
        load_kadid10k(settings)


def test_automatic_download_extracts_local_zip_fixture(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_synthetic_kadid(source_root, references=3, images_per_reference=2)
    archive = tmp_path / "kadid-fixture.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in source_root.rglob("*"):
            if path.is_file():
                handle.write(path, Path("kadid10k") / path.relative_to(source_root))
    settings = _settings(tmp_path / "destination")
    settings.download = True
    settings.download_url = archive.resolve().as_uri()
    settings.download_filename = "kadid-fixture.zip"
    report = ensure_kadid10k_dataset(settings)
    assert report["action"] == "download"
    inspection = inspect_kadid10k_root(settings.data_root)
    assert inspection["has_dmos_csv"]
    assert inspection["distorted_image_count"] == 6
    assert not (settings.data_root / "_downloads").exists()


def test_cache_identity_guards_dataset_split_scale_and_pixel_budget(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.resolved_annotations_file = str(tmp_path / "dmos.csv")
    settings.split_digest = "reference-split"
    identity = cache_identity(settings)
    assert identity["split_unit"] == "reference_image"
    assert identity["quality_score_min"] == 1.0
    assert identity["quality_score_max"] == 5.0
    assert identity["processor_max_pixels"] == 37632
    first_digest = cache_identity_digest(settings)
    first_root = precompute_cache_root(settings)
    settings.output_dir = tmp_path / "different-run"
    assert cache_identity_digest(settings) == first_digest
    assert precompute_cache_root(settings) == first_root
    settings.processor_max_pixels = 25600
    assert cache_identity_digest(settings) != first_digest


def test_teacher_cache_metadata_contains_kadid_identity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.resolved_annotations_file = str(tmp_path / "dmos.csv")
    settings.split_digest = "reference-split"
    settings.vision_depth = 24
    settings.vision_hidden_size = 1024
    settings.text_depth = 28
    settings.text_hidden_size = 2048
    settings.deepstack_visual_indexes = (5, 11, 17)
    metadata = expected_metadata("train", 20, settings, model=None)
    assert metadata["dataset"] == "kadid10k_mos"
    assert metadata["split_unit"] == "reference_image"
    assert metadata["target_scale"] == [0.0, 1.0]
    assert metadata["original_target_scale"] == [1.0, 5.0]
    assert metadata["processor_min_pixels"] == 37632


def test_metrics_are_reported_on_original_one_to_five_scale() -> None:
    targets = torch.tensor([0.0, 0.5, 1.0])
    predictions = torch.tensor([0.25, 0.5, 0.75])
    report = score_metrics(predictions, targets)
    assert report["mae"] == pytest.approx(2.0 / 3.0)
    assert report["prediction_min_original"] == pytest.approx(2.0)
    assert report["prediction_max_original"] == pytest.approx(4.0)
    assert report["target_scale_min"] == 1.0
    assert report["target_scale_max"] == 5.0


def test_norm_in_norm_is_affine_invariant_but_small_auxiliary_weight() -> None:
    targets = torch.tensor([0.05, 0.25, 0.45, 0.75, 0.95])
    identical = norm_in_norm_loss(targets.clone(), targets)
    affine = norm_in_norm_loss(2.5 * targets + 3.0, targets)
    reversed_loss = norm_in_norm_loss(1.0 - targets, targets)
    assert identical.item() == pytest.approx(0.0, abs=1e-7)
    assert affine.item() == pytest.approx(0.0, abs=1e-6)
    assert reversed_loss > affine
    settings = load_settings(CONFIGS / "kadid10k_mos.json")
    assert settings.loss_norm_in_norm_weight == pytest.approx(0.1)
    assert settings.loss_norm_in_norm_weight < settings.loss_regression_weight


def test_ranking_loss_prefers_correct_order_and_backpropagates() -> None:
    targets = torch.tensor([0.05, 0.25, 0.70, 0.95])
    correct = targets.clone().requires_grad_(True)
    reversed_scores = (1.0 - targets).requires_grad_(True)
    correct_loss = pairwise_ranking_loss(correct, targets)
    reversed_loss = pairwise_ranking_loss(reversed_scores, targets)
    assert correct_loss < reversed_loss
    reversed_loss.backward()
    assert reversed_scores.grad is not None
    assert torch.isfinite(reversed_scores.grad).all()


def test_moe16_geometry_and_parameter_budget_are_preserved() -> None:
    settings = load_settings(CONFIGS / "kadid10k_mos.json")
    geometry = MoEGeometry()
    geometry.validate()
    assert geometry.outer_padding == 20
    assert geometry.expert_gap == 30
    assert geometry.active_aperture == Aperture(20, 1006, 20, 1006)
    vision = HomogeneousMoEOpticalCore(1024, 224, settings)
    language = HomogeneousMoEOpticalCore(2048, 224, settings)
    vision_report = vision.parameter_breakdown()
    language_report = language.parameter_breakdown()
    assert vision_report["expert_phase_parameters"] == 16 * 4 * 224 * 224
    assert vision_report["global_phase_parameters"] == 986 * 986
    assert vision_report["optical_phase_parameters"] == 4_183_460
    assert vision_report["adapter_parameters"] == 460_448
    assert language_report["adapter_parameters"] == 920_224
    head_parameters = sum(
        parameter.numel() for parameter in build_head(settings, 2048).parameters()
    )
    total = (
        vision_report["trainable_parameters"]
        + language_report["trainable_parameters"]
        + head_parameters
    )
    assert total == 9_760_041


def _encoder(hidden_size: int = 8) -> HomogeneousMoEOpticalCore:
    module = HomogeneousMoEOpticalCore.__new__(HomogeneousMoEOpticalCore)
    nn.Module.__init__(module)
    module.hidden_size = hidden_size
    module.max_tokens = 224
    module.geometry = MoEGeometry()
    module.input_adapter = nn.Linear(hidden_size, 224)
    module.input_norm = nn.LayerNorm(224)
    module.nonnegative = nn.Softplus()
    module.amplitude_slm_weight_domain = "amplitude"
    module.amplitude_slm_input_normalization = "none"
    module.amplitude_phase_relay = "ideal_4f_identity"
    module.register_buffer(
        "expert_canvas_indices",
        aperture_linear_indices(
            module.geometry.canvas_size,
            module.geometry.expert_apertures,
        ),
        persistent=False,
    )
    module.last_input_fields = None
    module.last_routing = {}
    module.last_amplitude_slm_canvas = None
    module.last_stage_fields = []
    module.last_detector_intensity = None
    module.last_detector_readout = None
    module.capture_intermediate_fields = False
    module.capture_sample_count = 1
    return module


def test_token_rows_are_nonnegative_zero_padded_and_overflow_is_explicit() -> None:
    encoder = _encoder()
    field = encoder.encode_groups([torch.randn(60, 8)])
    assert field.shape == (1, 224, 224)
    assert torch.all(field >= 0)
    assert torch.count_nonzero(field[:, 60:]) == 0
    with pytest.raises(RuntimeError, match="visual token count 225"):
        encoder.encode_groups([torch.randn(225, 8)])


def test_language_overflow_is_explicit() -> None:
    language = LanguageDeepStackHomogeneousMoE.__new__(
        LanguageDeepStackHomogeneousMoE
    )
    nn.Module.__init__(language)
    language.core = SimpleNamespace(max_tokens=224)
    with pytest.raises(RuntimeError, match="language sequence length 225"):
        language.set_attention_mask(torch.ones(1, 225))


def test_electronic_top4_router_is_sparse_and_differentiable() -> None:
    router = ElectronicAmplitudeRouter(MoEGeometry(), 4, 4, 1.0)
    fields = torch.rand(2, 224, 224, requires_grad=True)
    output = router(fields)
    assert output["weights"].shape == (2, 16)
    assert torch.equal((output["weights"] > 0).sum(1), torch.tensor([4, 4]))
    assert torch.allclose(output["weights"].sum(1), torch.ones(2))
    assert output["phase_prompt_used"] is False
    (output["weights"].square().sum() + output["balance_loss"]).backward()
    assert router.router.gate.weight.grad is not None


def test_detector_readout_has_nonzero_phase_gradient() -> None:
    settings = SimpleNamespace(
        detector_output_size=8,
        detector_layernorm_scope="per_token",
        detector_layernorm_eps=1e-5,
        detector_layernorm_affine=False,
        detector_nonlinearity="relu",
    )
    geometry = SimpleNamespace(detector_aperture=Aperture(2, 14, 2, 14))
    readout = FullPlaneReadout(geometry, settings)
    phase = PhaseLayer(16, parameterization="unconstrained", init="small_normal")
    propagation = AngularSpectrumPropagator(
        wavelength_m=532e-9,
        pixel_size_m=16e-6,
        grid_size=16,
        distance_m=0.1,
    )
    torch.manual_seed(9)
    amplitude = torch.rand(2, 16, 16)
    values, detector_roi = readout(
        propagation(phase(amplitude.to(torch.complex64)))
    )
    assert values.shape == (2, 8, 8)
    assert detector_roi.shape == (2, 12, 12)
    weighted_loss = (
        values * torch.linspace(0.1, 1.0, 8)[None, None, :]
    ).mean()
    weighted_loss.backward()
    assert phase.raw_phase.grad is not None
    assert torch.isfinite(phase.raw_phase.grad).all()
    assert torch.count_nonzero(phase.raw_phase.grad) > 0


def test_regression_head_shape_and_backward() -> None:
    settings = load_settings(CONFIGS / "kadid10k_mos.json")
    head = build_head(settings, 2048)
    prediction = head(torch.randn(4, 2048))
    assert prediction.shape == (4,)
    torch.nn.functional.smooth_l1_loss(
        prediction,
        torch.rand(4),
        beta=settings.smooth_l1_beta,
    ).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
