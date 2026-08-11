from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.analyze_gallery_coverage import (
    select_additional_gallery_samples,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    OpticalRetrievalReadout,
    official_mrl_embedding,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline import (
    bin_ccd_superpixels,
    load_captured_intensity,
    load_hardware_config,
    select_samples,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_finetune import (
    _epoch_batches,
    _teacher_store_for_adaptation,
    configure_downstream_trainability,
    prototype_retrieval_objective,
    select_adaptation_rows,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_automation import (
    load_automation_config,
    normalized_ccd_comparison,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    encode_amplitude_uint8,
    encode_phase_uint8,
    export_centered_bmp,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.geometry import (
    Aperture,
    MoEGeometry,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    FullPlaneReadout,
    HomogeneousMoEOpticalCore,
    LanguageDeepStackHomogeneousMoE,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    PhaseLayer,
    SquareDetectionLayerNormReload,
    phase_dc_loss,
    phase_dc_statistics,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GrocerySample,
    prepare_grocery_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import (
    evaluate_embeddings,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    PKBatchSampler,
    _build_optimizer,
    _phase_focus_epoch,
    _phase_motion_statistics,
    _phase_reference,
    _set_phase_focus_trainability,
    embedding_distillation_loss,
    gallery_retrieval_logits,
    gallery_retrieval_loss,
    initialize_parameter_ema,
    relational_embedding_distillation_loss,
    update_parameter_ema,
    use_parameter_ema,
    retrieval_ranking_sums,
    select_gallery_items_for_queries,
    supervised_contrastive_loss,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_main_config_has_ten_packaged_skus() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "archive" / "historical_moe16_best.yaml"
    )
    assert len(settings.selected_skus) == 10
    assert settings.embedding_dim == 64
    assert settings.gallery_aggregation == "mean_prototype"
    assert settings.instruction == (
        "Represent this product image for image-to-image product retrieval."
    )
    assert settings.expert_layers == 1
    assert settings.vision_tap_stages == (1,)
    assert settings.output_dir.name == "historical_moe16_rerun"
    assert settings.num_experts == 16 and settings.expert_grid_rows == 4


def test_release_configs_are_small_and_explicit() -> None:
    release = sorted(path.name for path in (EXPERIMENT / "configs" / "release").glob("*.yaml"))
    archive = sorted(path.name for path in (EXPERIMENT / "configs" / "archive").glob("*.yaml"))
    assert release == [
        "hardware_moe4.yaml",
        "model_moe4.yaml",
        "stage1_grocery31_pretrain.yaml",
        "stage2_grocery10_finetune.yaml",
    ]
    assert archive == ["historical_moe16_best.yaml"]


def test_phase_learning_rate_gets_an_independent_optimizer_group() -> None:
    class Core(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_phase = nn.Parameter(torch.zeros(2, 2))
            self.input_adapter = nn.Linear(2, 2)
            self.input_norm = nn.LayerNorm(2)
            self.output_adapter = nn.Linear(2, 2)
            self.router = nn.Linear(2, 2)
            self.other = nn.Parameter(torch.ones(1))

    class Surrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = Core()

    class Replacement:
        def __init__(self) -> None:
            self.vision_surrogate = Surrogate()
            self.language_surrogate = Surrogate()

        def trainable_parameters(self):
            return list(self.vision_surrogate.parameters()) + list(
                self.language_surrogate.parameters()
            )

    replacement = Replacement()
    readout = OpticalRetrievalReadout(2, 2)
    settings = SimpleNamespace(
        learning_rate=1.0e-5,
        router_learning_rate=2.0e-5,
        phase_learning_rate=1.0e-6,
        adapter_learning_rate=3.0e-5,
        readout_learning_rate=4.0e-5,
        weight_decay=0.0,
    )
    optimizer, _ = _build_optimizer(replacement, readout, settings)
    groups = {
        group["group_name"]: group
        for group in optimizer.param_groups
    }
    assert groups["student_base"]["lr"] == 1.0e-5
    assert groups["optical_adapters"]["lr"] == 3.0e-5
    assert groups["retrieval_readout"]["lr"] == 4.0e-5
    assert groups["routers"]["lr"] == 2.0e-5
    assert groups["optical_phases"]["lr"] == 1.0e-6
    phase_ids = {
        id(replacement.vision_surrogate.core.raw_phase),
        id(replacement.language_surrogate.core.raw_phase),
    }
    assert {id(value) for value in groups["optical_phases"]["params"]} == phase_ids

    _set_phase_focus_trainability(optimizer, True)
    assert groups["optical_phases"]["lr"] == pytest.approx(1.0e-6)
    assert all(
        parameter.requires_grad
        for parameter in groups["optical_phases"]["params"]
    )
    assert all(
        not parameter.requires_grad
        for name, group in groups.items()
        if name != "optical_phases"
        for parameter in group["params"]
    )
    _set_phase_focus_trainability(optimizer, False)
    assert all(
        parameter.requires_grad
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_phase_focus_schedule_and_motion_are_explicit() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "release" / "model_moe4.yaml"
    )
    assert settings.phase_focus_enabled
    assert not _phase_focus_epoch(settings, 5)
    assert _phase_focus_epoch(settings, 6)
    assert not _phase_focus_epoch(settings, 7)
    assert _phase_focus_epoch(settings, 8)
    assert not _phase_focus_epoch(settings, 9)
    assert settings.phase_learning_rate == pytest.approx(4.0e-3)
    assert settings.adapter_learning_rate == pytest.approx(2.0e-4)
    assert settings.readout_learning_rate == pytest.approx(5.0e-4)
    assert not settings.transformer_residual_enabled
    assert not settings.phase_dc_enabled
    assert settings.lambda_phase_dc == pytest.approx(0.0)
    assert settings.phase_dc_start_epoch == 1
    assert settings.phase_init == "zeros"
    assert settings.phase_init_std == pytest.approx(0.0)
    assert settings.k_space_constraint_enabled
    assert settings.theta_max_deg == pytest.approx(0.65)
    assert settings.interlayer_detector_integration_factor == 2
    assert settings.optimizer_steps_per_epoch == 100

    phase_groups = {
        "vision_expert": [nn.Parameter(torch.zeros(2, 2))],
        "vision_global": [nn.Parameter(torch.zeros(2, 2))],
        "language_expert": [nn.Parameter(torch.zeros(2, 2))],
        "language_global": [nn.Parameter(torch.zeros(2, 2))],
    }
    reference = _phase_reference(phase_groups)
    unchanged = _phase_motion_statistics(phase_groups, reference, reference)
    assert unchanged["phase_physical_std_rad"] == pytest.approx(0.0, abs=1.0e-6)
    assert unchanged["phase_delta_run_rms_rad"] == pytest.approx(0.0)
    with torch.no_grad():
        phase_groups["vision_expert"][0][0, 0] = 0.5
    changed = _phase_motion_statistics(phase_groups, reference, reference)
    assert changed["vision_expert_phase_delta_run_rms_rad"] > 0.0
    assert changed["phase_delta_run_rms_rad"] > 0.0


def test_phase_dc_loss_is_per_plane_and_has_gradient_after_symmetry_breaking() -> None:
    module = nn.Sequential(
        PhaseLayer(8, init="small_normal", init_std=0.2),
        PhaseLayer(8, init="small_normal", init_std=0.2),
    )
    loss = phase_dc_loss(module)
    assert loss.ndim == 0
    assert 0.0 <= float(loss.detach()) <= 1.0 + 1.0e-6
    loss.backward()
    assert all(layer.raw_phase.grad is not None for layer in module)
    assert all(torch.count_nonzero(layer.raw_phase.grad) > 0 for layer in module)
    stats = phase_dc_statistics(module)
    assert stats["phase_dc_plane_count"] == 2
    assert stats["phase_dc_current_loss"] == pytest.approx(float(loss.detach()))


def test_uniform_phase_dc_loss_exposes_zero_gradient_stationary_point() -> None:
    layer = PhaseLayer(8, init="zeros")
    loss = phase_dc_loss(layer)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(1.0, abs=1.0e-6)
    assert torch.allclose(layer.raw_phase.grad, torch.zeros_like(layer.raw_phase))


def test_phase_dc_loss_accepts_non_module_replacement_wrapper() -> None:
    class Wrapper:
        def __init__(self) -> None:
            self.vision_surrogate = nn.Sequential(
                PhaseLayer(4, init="small_normal", init_std=0.2)
            )
            self.language_surrogate = nn.Sequential(
                PhaseLayer(4, init="small_normal", init_std=0.2)
            )

    wrapper = Wrapper()
    loss = phase_dc_loss(wrapper)
    loss.backward()
    assert torch.isfinite(loss)
    assert phase_dc_statistics(wrapper)["phase_dc_plane_count"] == 2
    for surrogate in (wrapper.vision_surrogate, wrapper.language_surrogate):
        assert surrogate[0].raw_phase.grad is not None


def test_relational_kd_matches_pairwise_teacher_geometry() -> None:
    teacher = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        dim=-1,
    )
    identical = relational_embedding_distillation_loss(teacher.clone(), teacher)
    distorted = relational_embedding_distillation_loss(
        teacher[[0, 2, 1]], teacher
    )
    assert float(identical) == pytest.approx(0.0, abs=1.0e-8)
    assert float(distorted) > 0.0


def test_parameter_ema_updates_and_restores_live_weights() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 3.0]))
    ema = initialize_parameter_ema([parameter])
    with torch.no_grad():
        parameter.copy_(torch.tensor([3.0, 7.0]))
    update_parameter_ema(ema, [parameter], decay=0.5)
    assert torch.equal(ema[0], torch.tensor([2.0, 5.0]))
    live = parameter.detach().clone()
    with use_parameter_ema([parameter], ema):
        assert torch.equal(parameter.detach(), torch.tensor([2.0, 5.0]))
    assert torch.equal(parameter.detach(), live)


def test_canonical_best_and_latest_configs_are_explicit() -> None:
    best = load_settings(
        EXPERIMENT / "configs" / "archive" / "historical_moe16_best.yaml"
    )
    latest = load_settings(
        EXPERIMENT / "configs" / "release" / "model_moe4.yaml"
    )
    assert best.num_experts == 16 and best.top_k == 4
    assert best.epochs == 40 and best.ema_decay == pytest.approx(0.99)
    assert best.phase_learning_rate == pytest.approx(1.0e-3)
    assert best.crop_scale_min == pytest.approx(0.75)
    assert latest.num_experts == 4 and latest.top_k == 2
    assert latest.pixel_pitch_um == pytest.approx(16.0)
    assert latest.interlayer_detector_integration_factor == 2


def test_slm_encoding_preserves_relative_amplitude_and_wraps_phase() -> None:
    encoded, metadata = encode_amplitude_uint8(torch.tensor([[0.0, 1.0, 2.0]]))
    assert encoded.tolist() == [[0, 128, 255]]
    assert metadata["normalization_divisor"] == pytest.approx(2.0)
    phase = encode_phase_uint8(
        torch.tensor([[0.0, torch.pi, 2.0 * torch.pi]])
    )
    assert phase[0, 0].item() == 0
    assert phase[0, 1].item() in {127, 128}
    assert phase[0, 2].item() == 0


def test_phase_bmp_vertical_flip_is_exact(tmp_path: Path) -> None:
    source = torch.tensor([[0.0, 0.0], [torch.pi, torch.pi]])
    report = export_centered_bmp(
        source,
        tmp_path / "flipped_phase.bmp",
        value_type="phase",
        scale_factor=1,
        slm_width=2,
        slm_height=2,
        flip_vertical=True,
    )
    pixels = torch.from_numpy(__import__("numpy").array(Image.open(tmp_path / "flipped_phase.bmp")).copy())
    assert torch.all(pixels[0] >= 127)
    assert torch.all(pixels[1] == 0)
    assert report["flip_vertical_before_export"] is True


def test_percentile_amplitude_encoding_ignores_padding_and_clips_outlier() -> None:
    amplitude = torch.tensor([[0.0] * 100 + [1.0] * 99 + [100.0]])
    encoded, metadata = encode_amplitude_uint8(
        amplitude,
        mode="positive_percentile",
        percentile=99.0,
    )
    assert metadata["normalization_divisor"] < 100.0
    assert metadata["raw_nonzero_ratio"] == pytest.approx(0.5)
    assert metadata["raw_clipped_ratio"] > 0.0
    assert int(encoded[0, -1]) == 255
    assert int(encoded[0, 100]) > 2


def test_grocery_active_plane_bmp_uses_scale_one_and_exact_centering(
    tmp_path: Path,
) -> None:
    source = torch.ones(986, 986)
    amplitude = export_centered_bmp(
        source,
        tmp_path / "amplitude.bmp",
        value_type="amplitude",
        scale_factor=1,
        slm_width=1920,
        slm_height=1080,
    )
    phase = export_centered_bmp(
        torch.zeros_like(source),
        tmp_path / "phase.bmp",
        value_type="phase",
        scale_factor=1,
        slm_width=1920,
        slm_height=1200,
    )
    assert Image.open(tmp_path / "amplitude.bmp").mode == "L"
    assert Image.open(tmp_path / "amplitude.bmp").size == (1920, 1080)
    assert Image.open(tmp_path / "phase.bmp").size == (1920, 1200)
    assert amplitude["active_bounds_xyxy"] == [467, 47, 1453, 1033]
    assert amplitude["center_padding_lrtb"] == [467, 467, 47, 47]
    assert phase["active_bounds_xyxy"] == [467, 107, 1453, 1093]
    assert phase["center_padding_lrtb"] == [467, 467, 107, 107]
    with pytest.raises(ValueError, match="exceeds SLM"):
        export_centered_bmp(
            source,
            tmp_path / "wrong_scale.bmp",
            value_type="phase",
            scale_factor=2,
            slm_width=1920,
            slm_height=1200,
        )


def test_gallery_coverage_selection_is_train_only_and_deterministic(
    tmp_path: Path,
) -> None:
    names = ("a", "b")
    train = tuple(
        GrocerySample(
            f"train-{sku}-{index}",
            tmp_path / f"train-{sku}-{index}.jpg",
            sku,
            names[sku],
            sku,
            "train",
            "train",
            False,
        )
        for sku in range(2)
        for index in range(3)
    )
    galleries = tuple(
        GrocerySample(
            f"gallery-{sku}",
            tmp_path / f"gallery-{sku}.jpg",
            sku,
            names[sku],
            sku,
            "gallery",
            "iconic",
            True,
        )
        for sku in range(2)
    )
    train_embeddings = torch.tensor(
        [
            [0.99, 0.01],
            [0.80, 0.20],
            [0.60, 0.40],
            [0.01, 0.99],
            [0.20, 0.80],
            [0.40, 0.60],
        ]
    )
    gallery_embeddings = torch.eye(2)
    selected, rows = select_additional_gallery_samples(
        train,
        train_embeddings,
        galleries,
        gallery_embeddings,
        per_sku=2,
    )
    assert [sample.sample_id for sample in selected] == [
        "train-0-0",
        "train-0-1",
        "train-1-0",
        "train-1-1",
    ]
    assert all(row["selection_source"] == "train_only" for row in rows)
    assert all(row["test_used_for_selection"] is False for row in rows)


def test_student_has_one_expert_stage_plus_one_global_phase() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "archive" / "historical_moe16_best.yaml"
    )
    vision = HomogeneousMoEOpticalCore(1024, 224, settings)
    language = HomogeneousMoEOpticalCore(2048, 224, settings)
    assert len(vision.expert_layers) == 1
    assert len(language.expert_layers) == 1
    assert len(vision.interlayer_conversions) == 1
    assert len(language.interlayer_conversions) == 1
    vision_report = vision.parameter_breakdown()
    language_report = language.parameter_breakdown()
    assert vision_report["expert_phase_parameters"] == 16 * 224 * 224
    assert language_report["expert_phase_parameters"] == 16 * 224 * 224
    assert vision_report["global_phase_parameters"] == 986 * 986
    assert language_report["global_phase_parameters"] == 986 * 986
    assert vision_report["optical_phase_parameters"] == 1_775_012
    assert language_report["optical_phase_parameters"] == 1_775_012
    total_with_readout = (
        vision_report["trainable_parameters"]
        + language_report["trainable_parameters"]
        + sum(
            parameter.numel()
            for parameter in OpticalRetrievalReadout(224, 64).parameters()
        )
    )
    assert total_with_readout == 4_951_848


def test_moe4_superpixel2_geometry_and_phase_parameter_count() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "release" / "model_moe4.yaml"
    )
    geometry = MoEGeometry(
        settings.canvas_size,
        settings.active_size,
        settings.expert_size,
        settings.expert_pitch,
        settings.num_experts,
        settings.expert_grid_rows,
        settings.expert_grid_cols,
    )
    geometry.validate()
    assert geometry.outer_padding == 20
    assert geometry.expert_gap == 30
    assert geometry.footprint_width == geometry.footprint_height == 478
    assert len(geometry.expert_apertures) == 4
    assert settings.top_k == 2
    assert settings.pixel_pitch_um == pytest.approx(16.0)
    core = HomogeneousMoEOpticalCore(1024, 224, settings)
    report = core.parameter_breakdown()
    assert report["expert_phase_parameters"] == 4 * 224 * 224
    assert report["global_phase_parameters"] == 478 * 478
    assert report["optical_phase_parameters"] == 429_188


def test_two_by_two_ccd_binning_is_exact_block_reduction() -> None:
    physical = torch.tensor(
        [
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 7.0, 6.0, 8.0],
            [9.0, 11.0, 10.0, 12.0],
            [13.0, 15.0, 14.0, 16.0],
        ]
    )
    assert torch.equal(
        bin_ccd_superpixels(physical, 2, "mean"),
        torch.tensor([[4.0, 5.0], [12.0, 13.0]]),
    )
    assert torch.equal(
        bin_ccd_superpixels(physical, 2, "sum"),
        torch.tensor([[16.0, 20.0], [48.0, 52.0]]),
    )


def test_oeo_detector_integration_is_exact_two_by_two_zero_order_hold() -> None:
    conversion = SquareDetectionLayerNormReload(
        4,
        [Aperture(0, 4, 0, 4)],
        1.0e-5,
        "relu",
        detector_integration_factor=2,
    )
    conversion.set_intermediate_capture(True)
    intensity = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    conversion.forward_intensity(intensity)
    expected = torch.tensor(
        [[2.5, 2.5, 4.5, 4.5], [2.5, 2.5, 4.5, 4.5],
         [10.5, 10.5, 12.5, 12.5], [10.5, 10.5, 12.5, 12.5]]
    )
    assert torch.equal(conversion.last_input_intensity[0], expected)


def test_official_dataset_split_and_no_leakage(tmp_path: Path) -> None:
    root = tmp_path / "GroceryStoreDataset"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    base_settings = load_settings(
        EXPERIMENT / "configs" / "archive" / "historical_moe16_best.yaml"
    )
    headers = [
        "Class Name (str)",
        "Class ID (int)",
        "Coarse Class Name (str)",
        "Coarse Class ID (int)",
        "Iconic Image Path (str)",
        "Product Description Path (str)",
    ]
    rows = []
    split_lines = {"train": [], "test": []}
    for index, name in enumerate(base_settings.selected_skus):
        iconic = f"iconic/{name}.jpg"
        train = f"train/{name}/train.jpg"
        test = f"test/{name}/test.jpg"
        _image(dataset / iconic, color=(index * 20 % 255, 30, 60))
        _image(dataset / train, color=(30, index * 20 % 255, 60))
        _image(dataset / test, color=(30, 60, index * 20 % 255))
        rows.append([name, index, "Package", 1, "/" + iconic, "unused.txt"])
        split_lines["train"].append(f"{train}, {index}, 1\n")
        split_lines["test"].append(f"{test}, {index}, 1\n")
    with (dataset / "classes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    (dataset / "train.txt").write_text("".join(split_lines["train"]), encoding="utf-8")
    (dataset / "val.txt").write_text("", encoding="utf-8")
    (dataset / "test.txt").write_text("".join(split_lines["test"]), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "base_config": str(
                    EXPERIMENT / "configs" / "archive" / "historical_moe16_best.yaml"
                ),
                "dataset": {"dataset_root": str(root), "download": False},
                "output_dir": str(tmp_path / "run"),
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    bundle = prepare_grocery_subset(settings, persist=True)
    assert len(bundle.train_samples) == 10
    assert len(bundle.test_samples) == 10
    assert len(bundle.gallery_samples) == 10
    assert {
        sample.image_path for sample in bundle.train_samples
    }.isdisjoint(sample.image_path for sample in bundle.test_samples)
    assert (settings.output_dir / "manifests" / "grocery10_subset.csv").is_file()


def test_official_mrl_embedding_shape_and_normalization() -> None:
    hidden = torch.randn(3, 7, 2048)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    )
    output = official_mrl_embedding(hidden, mask, 64)
    expected_rows = torch.stack([hidden[0, 2, :64], hidden[1, 4, :64], hidden[2, 6, :64]])
    assert output.shape == (3, 64)
    assert torch.allclose(output, torch.nn.functional.normalize(expected_rows, dim=-1))
    assert torch.allclose(output.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_optical_readout_is_signed_linear_then_l2() -> None:
    head = OpticalRetrievalReadout(224, 64)
    output = head(torch.rand(4, 224))
    assert output.shape == (4, 64)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert not any(
        isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Softmax))
        for module in head.modules()
    )
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_final_detector_features_are_nonnegative_last_valid_rows() -> None:
    language = LanguageDeepStackHomogeneousMoE.__new__(
        LanguageDeepStackHomogeneousMoE
    )
    nn.Module.__init__(language)
    detector = torch.rand(2, 224, 224)
    language.core = SimpleNamespace(
        current_detector_readout=detector,
        readout=SimpleNamespace(output_size=224),
    )
    language.lengths = [3, 5]
    output = language.retrieval_detector_features()
    assert output.shape == (2, 224)
    assert torch.equal(output[0], detector[0, 2])
    assert torch.equal(output[1], detector[1, 4])
    assert torch.all(output >= 0)


def test_square_law_detector_readout_is_nonnegative() -> None:
    geometry = SimpleNamespace(detector_aperture=Aperture(1, 7, 1, 7))
    settings = SimpleNamespace(
        detector_output_size=4,
        detector_layernorm_scope="per_token",
        detector_layernorm_eps=1e-5,
        detector_layernorm_affine=False,
        detector_nonlinearity="relu",
    )
    readout = FullPlaneReadout(geometry, settings)
    field = torch.complex(torch.randn(2, 8, 8), torch.randn(2, 8, 8))
    values, intensity = readout(field)
    assert values.shape == (2, 4, 4)
    assert intensity.shape == (2, 6, 6)
    assert torch.all(values >= 0)
    assert torch.all(intensity >= 0)


def test_measured_intensity_entry_points_do_not_square_twice() -> None:
    apertures = [Aperture(0, 2, 0, 2), Aperture(0, 2, 2, 4)]
    conversion = SquareDetectionLayerNormReload(
        4,
        apertures,
        1.0e-5,
        "relu",
        per_expert_enabled=True,
        elementwise_affine=False,
    )
    field = torch.complex(torch.rand(2, 4, 4), torch.rand(2, 4, 4))
    intensity = field.abs().square()
    selected = torch.ones(2, 2, dtype=torch.bool)
    weights = torch.full((2, 2), 0.5)
    conversion.set_intermediate_capture(True, sample_count=2)
    simulated = conversion(field, selected, weights)
    measured = conversion.forward_intensity(intensity, selected, weights)
    assert torch.allclose(simulated, measured, atol=1.0e-6)
    assert torch.all(measured.real >= 0)
    assert torch.count_nonzero(measured.imag) == 0
    assert conversion.last_input_complex_field is not None
    assert torch.equal(conversion.last_input_complex_field, field.to(torch.complex64).cpu())

    geometry = SimpleNamespace(detector_aperture=Aperture(1, 7, 1, 7))
    settings = SimpleNamespace(
        detector_output_size=4,
        detector_layernorm_scope="per_token",
        detector_layernorm_eps=1e-5,
        detector_layernorm_affine=False,
        detector_nonlinearity="relu",
    )
    readout = FullPlaneReadout(geometry, settings)
    regular, cropped = readout(field.new_zeros((2, 8, 8)).copy_(
        torch.complex(torch.rand(2, 8, 8), torch.rand(2, 8, 8))
    ))
    replay, replay_intensity = readout.forward_intensity(cropped)
    assert torch.allclose(regular, replay, atol=1.0e-6)
    assert torch.equal(cropped, replay_intensity)


def test_hardware_config_is_stage_first_and_uses_best_checkpoint() -> None:
    full = load_hardware_config(
        EXPERIMENT / "configs" / "release" / "hardware_moe4.yaml"
    )
    assert full.selection_mode == "full_dataset"
    assert full.checkpoint.name == "ema_last_checkpoint.pt"
    assert full.queries_per_sku == 10
    assert full.amplitude_slm_size == (1920, 1080)
    assert full.phase_slm_size == (1920, 1200)
    assert full.capture_binning_factor == 2
    assert full.capture_registration_mode == "nearest_resize"
    assert full.capture_flip_vertical and full.capture_flip_horizontal
    assert full.phase_flip_vertical
    assert not full.minimal_artifacts
    assert not full.copy_checkpoint_to_output
    assert full.amplitude_encoding_percentile == pytest.approx(95.0)
    assert full.amplitude_encoding_gamma == pytest.approx(0.65)


def test_ccd_nearest_registration_then_exact_superpixel_binning(tmp_path: Path) -> None:
    hardware = load_hardware_config(
        EXPERIMENT / "configs" / "release" / "hardware_moe4.yaml"
    )
    hardware = replace(
        hardware,
        output_dir=tmp_path,
        capture_flip_vertical=True,
        capture_flip_horizontal=True,
    )
    capture = tmp_path / "01_vision_expert" / "ccd_captured"
    capture.mkdir(parents=True)
    source = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    torch.save(source, capture / "sample.pt")
    runtime = SimpleNamespace(
        hardware=hardware,
        settings=SimpleNamespace(active_size=2),
    )
    actual = load_captured_intensity(
        runtime, "vision_expert", "sample", use_simulation=False
    )
    registered = F.interpolate(
        torch.flip(source, dims=(-2, -1))[None, None], size=(4, 4), mode="nearest"
    )[0, 0]
    expected = bin_ccd_superpixels(registered, 2, "mean")
    assert torch.equal(actual, expected)
    metadata = tmp_path / "01_vision_expert" / "registered_ccd" / "sample.json"
    assert metadata.is_file()
    assert '"flip_vertical_after_roi": true' in metadata.read_text(encoding="utf-8")
    assert '"flip_horizontal_after_roi": true' in metadata.read_text(encoding="utf-8")


class _AdaptationCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_adapter = nn.Linear(2, 2)
        self.input_norm = nn.LayerNorm(2)
        self.router = nn.Linear(2, 2)
        self.expert_layers = nn.ModuleList([nn.Linear(2, 2)])
        self.interlayer_conversions = nn.ModuleList([nn.Linear(2, 2)])
        self.global_phase = nn.Linear(2, 2)
        self.readout = nn.LayerNorm(2)
        self.output_adapter = nn.Linear(2, 2)


class _AdaptationSurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _AdaptationCore()


def test_final_ccd_adaptation_trains_only_downstream_electronics() -> None:
    vision = _AdaptationSurrogate()
    language = _AdaptationSurrogate()
    readout = nn.Linear(2, 2)
    runtime = SimpleNamespace(
        loaded=SimpleNamespace(model=nn.Linear(2, 2)),
        replacement=SimpleNamespace(
            vision_surrogate=vision, language_surrogate=language
        ),
        readout=readout,
    )
    parameters, report = configure_downstream_trainability(
        runtime, "language_global"
    )
    assert parameters
    assert all(parameter.requires_grad for parameter in language.core.readout.parameters())
    assert all(parameter.requires_grad for parameter in readout.parameters())
    assert all(not parameter.requires_grad for parameter in language.core.global_phase.parameters())
    assert all(not parameter.requires_grad for parameter in vision.parameters())
    assert {row["name"].split(".")[0] for row in report} == {
        "language",
        "retrieval_readout",
    }


def test_zero_kd_hardware_adaptation_does_not_require_teacher_cache() -> None:
    assert _teacher_store_for_adaptation(
        SimpleNamespace(), SimpleNamespace(lambda_kd=0.0)
    ) is None


def test_hardware_epoch_batches_cover_every_query_and_include_all_galleries() -> None:
    samples = {}
    rows = []
    for sku in range(3):
        gallery_id = f"g{sku}"
        samples[gallery_id] = SimpleNamespace(sku_index=sku)
        rows.append({"sample_id": gallery_id, "sample_key": gallery_id, "role": "gallery"})
        for index in range(5):
            sample_id = f"q{sku}_{index}"
            samples[sample_id] = SimpleNamespace(sku_index=sku)
            rows.append({"sample_id": sample_id, "sample_key": sample_id, "role": "query"})
    config = SimpleNamespace(skus_per_batch=3, samples_per_sku=2)
    batches = _epoch_batches(rows, samples, config, epoch=1)
    assert len(batches) == 3
    query_ids = [
        row["sample_id"]
        for batch in batches
        for row in batch
        if row["role"] == "query"
    ]
    assert sorted(query_ids) == sorted(
        row["sample_id"] for row in rows if row["role"] == "query"
    )
    for batch in batches:
        assert {samples[row["sample_id"]].sku_index for row in batch if row["role"] == "gallery"} == {0, 1, 2}


def test_hardware_epoch_batches_merge_single_sku_tail_without_duplication() -> None:
    samples = {}
    rows = []
    for sku, query_count in enumerate((5, 4, 1)):
        gallery_id = f"g{sku}"
        samples[gallery_id] = SimpleNamespace(sku_index=sku)
        rows.append({"sample_id": gallery_id, "sample_key": gallery_id, "role": "gallery"})
        for index in range(query_count):
            sample_id = f"q{sku}_{index}"
            samples[sample_id] = SimpleNamespace(sku_index=sku)
            rows.append({"sample_id": sample_id, "sample_key": sample_id, "role": "query"})
    batches = _epoch_batches(
        rows,
        samples,
        SimpleNamespace(skus_per_batch=3, samples_per_sku=2),
        epoch=1,
    )
    query_ids = [
        row["sample_id"]
        for batch in batches
        for row in batch
        if row["role"] == "query"
    ]
    expected = [row["sample_id"] for row in rows if row["role"] == "query"]
    assert sorted(query_ids) == sorted(expected)
    assert len(query_ids) == len(set(query_ids))
    for batch in batches:
        counts: dict[int, int] = {}
        for row in batch:
            sku = samples[row["sample_id"]].sku_index
            counts[sku] = counts.get(sku, 0) + 1
        assert len(counts) >= 2
        assert min(counts.values()) >= 2


def test_full_hardware_selection_exports_gallery_train_and_all_test() -> None:
    def sample(sample_id: str, split: str, gallery: bool = False):
        return SimpleNamespace(
            sample_id=sample_id,
            image_path=Path(f"/{sample_id}.png"),
            split=split,
            source_split=split,
            is_gallery=gallery,
            sku_index=0,
        )

    bundle = SimpleNamespace(
        gallery_samples=(sample("g", "gallery", True),),
        train_samples=(sample("t2", "train"), sample("t1", "train")),
        test_samples=(sample("q2", "test"), sample("q1", "test")),
    )
    hardware = SimpleNamespace(
        selection_mode="full_dataset", sample_limit=None
    )
    selected = select_samples(bundle, SimpleNamespace(), hardware)
    assert [(role, value.sample_id) for role, value in selected] == [
        ("gallery", "g"),
        ("train", "t1"),
        ("train", "t2"),
        ("query", "q1"),
        ("query", "q2"),
    ]

    test_only = SimpleNamespace(selection_mode="test_only", sample_limit=None)
    selected_test = select_samples(bundle, SimpleNamespace(), test_only)
    assert [(role, value.sample_id) for role, value in selected_test] == [
        ("gallery", "g"),
        ("query", "q1"),
        ("query", "q2"),
    ]


def test_hardware_adaptation_test_inclusion_is_explicit() -> None:
    rows = [
        {"sample_id": "g", "role": "gallery"},
        {"sample_id": "t", "role": "train"},
        {"sample_id": "q", "role": "query"},
    ]
    independent, independent_report = select_adaptation_rows(
        rows, include_test_split=False
    )
    assert [row["sample_id"] for row in independent] == ["g", "t"]
    assert independent[-1]["role"] == "query"
    assert independent[-1]["source_role"] == "train"
    assert independent_report["independent_test_evaluation"] is True

    transductive, transductive_report = select_adaptation_rows(
        rows, include_test_split=True
    )
    assert [row["sample_id"] for row in transductive] == ["g", "t", "q"]
    assert transductive_report["test_count_used_for_adaptation"] == 1
    assert transductive_report["independent_test_evaluation"] is False


def test_measured_gallery_prototype_objective_has_perfect_top1_and_backward() -> None:
    samples = []
    rows = []
    values = []
    for sku in range(3):
        for role in ("gallery", "query", "query"):
            samples.append(SimpleNamespace(sku_index=sku))
            rows.append({"role": role})
            values.append(torch.eye(3)[sku])
    raw = torch.stack(values).requires_grad_(True)
    embeddings = F.normalize(raw, dim=-1)
    loss, metrics = prototype_retrieval_objective(
        embeddings, rows, samples, temperature=0.07
    )
    assert metrics["query_count"] == 6
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top3"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_release_hardware_config_leaves_device_control_external() -> None:
    config_path = EXPERIMENT / "configs" / "release" / "hardware_moe4.yaml"
    automation = load_automation_config(config_path)
    assert automation.settle_delay_seconds == pytest.approx(0.040)
    assert automation.confirm_each_phase_mask
    assert automation.output_extension == ".npy"
    assert automation.camera == {}
    assert automation.amplitude_slm == {"driver": "manual"}
    assert automation.phase_slm == {"driver": "manual"}


def test_ccd_comparison_reports_zero_mse_and_unit_pcc_for_identical_fields() -> None:
    field = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    metrics = normalized_ccd_comparison(field, field, "mean")
    assert metrics["normalized_mse"] == pytest.approx(0.0)
    assert metrics["normalized_mae"] == pytest.approx(0.0)
    assert metrics["pcc"] == pytest.approx(1.0)

    scaled = normalized_ccd_comparison(field * 9.0, field, "mean")
    assert scaled["normalized_mse"] == pytest.approx(0.0, abs=1e-12)
    assert scaled["pcc"] == pytest.approx(1.0)


def test_pk_sampler_and_supervised_contrastive_backward(tmp_path: Path) -> None:
    samples = []
    for sku in range(3):
        for image_index in range(4):
            samples.append(
                GrocerySample(
                    f"{sku}:{image_index}",
                    tmp_path / f"{sku}_{image_index}.jpg",
                    sku,
                    f"sku{sku}",
                    sku,
                    "train",
                    "train",
                    False,
                )
            )
    sampler = PKBatchSampler(samples, p=3, k=2, seed=42)
    batch = next(iter(sampler))
    labels = torch.tensor([samples[index].sku_index for index in batch])
    assert len(batch) == 6
    assert sorted(torch.bincount(labels).tolist()) == [2, 2, 2]
    raw = torch.randn(6, 64, requires_grad=True)
    embedding = torch.nn.functional.normalize(raw, dim=-1)
    loss = supervised_contrastive_loss(embedding, labels, 0.07)
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_kd_loss_zero_for_identical_embeddings() -> None:
    values = torch.nn.functional.normalize(torch.randn(5, 64), dim=-1)
    assert embedding_distillation_loss(values, values).abs() < 1e-6


def test_gallery_retrieval_loss_uses_wrong_skus_as_negatives_and_backpropagates() -> None:
    raw_gallery = torch.eye(3, requires_grad=True)
    raw_query = (torch.eye(3) + 0.01 * torch.randn(3, 3)).requires_grad_()
    labels = torch.arange(3)
    good = gallery_retrieval_loss(
        raw_query, labels, raw_gallery, labels, temperature=0.07
    )
    bad = gallery_retrieval_loss(
        raw_query, labels, raw_gallery, labels.roll(1), temperature=0.07
    )
    assert good < bad
    good.backward()
    assert raw_query.grad is not None and torch.isfinite(raw_query.grad).all()
    assert raw_gallery.grad is not None and torch.isfinite(raw_gallery.grad).all()


def test_gallery_stop_gradient_keeps_query_gradient() -> None:
    raw_gallery = torch.eye(3, requires_grad=True)
    raw_query = torch.eye(3, requires_grad=True)
    labels = torch.arange(3)
    loss = gallery_retrieval_loss(
        raw_query,
        labels,
        raw_gallery,
        labels,
        temperature=0.15,
        stop_gradient_on_gallery=True,
    )
    loss.backward()
    assert raw_query.grad is not None and torch.isfinite(raw_query.grad).all()
    assert raw_gallery.grad is None


def test_gallery_selection_and_training_retrieval_metrics(tmp_path: Path) -> None:
    galleries = []
    queries = []
    for sku in range(4):
        sample = GrocerySample(
            f"g{sku}",
            tmp_path / f"g{sku}.jpg",
            sku,
            f"sku{sku}",
            sku,
            "gallery",
            "iconic",
            True,
        )
        galleries.append({"image": object(), "sample": sample, "dataset_index": sku})
        if sku in {1, 3}:
            queries.append(
                GrocerySample(
                    f"q{sku}",
                    tmp_path / f"q{sku}.jpg",
                    sku,
                    f"sku{sku}",
                    sku,
                    "train",
                    "train",
                    False,
                )
            )
    selected = select_gallery_items_for_queries(galleries, queries)
    assert [item["sample"].sku_index for item in selected] == [1, 3]

    query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery_embeddings = query_embeddings.clone()
    labels = torch.tensor([1, 3])
    logits, targets = gallery_retrieval_logits(
        query_embeddings,
        labels,
        gallery_embeddings,
        labels,
        temperature=0.15,
    )
    values = retrieval_ranking_sums(logits, targets)
    assert values == {
        "top1_correct": 2.0,
        "top3_correct": 2.0,
        "reciprocal_rank_sum": 2.0,
        "query_count": 2.0,
    }


def test_retrieval_metrics_top1_top3_and_mrr(tmp_path: Path) -> None:
    class_names = ("a", "b", "c")
    gallery = torch.eye(3)
    query = torch.eye(3)
    galleries = [
        GrocerySample(f"g{i}", tmp_path / f"g{i}.jpg", i, name, i, "gallery", "iconic", True)
        for i, name in enumerate(class_names)
    ]
    queries = [
        GrocerySample(f"q{i}", tmp_path / f"q{i}.jpg", i, name, i, "test", "test", False)
        for i, name in enumerate(class_names)
    ]
    result = evaluate_embeddings(
        query, queries, gallery, galleries, class_names, "mean_prototype", system_name="test"
    )
    assert result.metrics["top1_retrieval_accuracy"] == 1.0
    assert result.metrics["top3_retrieval_accuracy"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert torch.equal(result.confusion, torch.eye(3, dtype=torch.long))


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
