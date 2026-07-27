from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.analyze_gallery_coverage import (
    select_additional_gallery_samples,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    OpticalRetrievalReadout,
    official_mrl_embedding,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.geometry import (
    Aperture,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    FullPlaneReadout,
    HomogeneousMoEOpticalCore,
    LanguageDeepStackHomogeneousMoE,
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
    embedding_distillation_loss,
    gallery_retrieval_logits,
    gallery_retrieval_loss,
    retrieval_ranking_sums,
    select_gallery_items_for_queries,
    supervised_contrastive_loss,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_main_config_has_ten_packaged_skus() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "grocery10.yaml")
    assert len(settings.selected_skus) == 10
    assert settings.embedding_dim == 64
    assert settings.gallery_aggregation == "mean_prototype"
    assert settings.instruction == (
        "Represent this product image for image-to-image product retrieval."
    )
    assert settings.expert_layers == 1
    assert settings.vision_tap_stages == (1,)
    assert settings.output_dir == (
        EXPERIMENT / "runs" / "qwen3_vl_embedding_2b_grocery10_optical_retrieval"
    ).resolve()


def test_smoke_output_is_also_inside_experiment() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "grocery10_smoke.yaml")
    assert settings.output_dir == (
        EXPERIMENT
        / "runs"
        / "qwen3_vl_embedding_2b_grocery10_optical_retrieval_smoke"
    ).resolve()


def test_grocery31_pretrain_config_is_valid_and_keeps_forward_batch_bounded() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "grocery31_pretrain.yaml")
    assert len(settings.selected_skus) == 31
    assert len(set(settings.selected_skus)) == 31
    assert settings.pk_skus_per_batch == 10
    assert settings.batch_size == 30
    # The implementation appends one gallery image for only each selected SKU.
    assert settings.batch_size + settings.pk_skus_per_batch == 40
    assert settings.dataset_variant == "grocery31"
    assert settings.subset_manifest_path.name == "grocery31_subset.csv"


def test_replaced_ten_sku_finetune_config() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "grocery10_replaced_finetune.yaml"
    )
    assert len(settings.selected_skus) == 10
    assert "Garant-Ecological-Standard-Milk" not in settings.selected_skus
    assert "Bravo-Apple-Juice" not in settings.selected_skus
    assert "God-Morgon-Apple-Juice" in settings.selected_skus
    assert "Tropicana-Mandarin-Morning" in settings.selected_skus
    assert settings.epochs == 50
    assert not settings.resume_optimizer_state
    assert settings.dataset_variant == "grocery10"


def test_epoch141_generalization_continuation_config() -> None:
    settings = load_settings(
        EXPERIMENT
        / "configs"
        / "grocery10_replaced_continue_epoch141_augmented_kd.yaml"
    )
    assert settings.epochs == 100
    assert 141 + settings.epochs == 241
    assert settings.learning_rate == 1.0e-5
    assert settings.router_learning_rate == 2.0e-5
    assert settings.lambda_kd == 8.0
    assert settings.lambda_gallery == 0.25
    assert settings.crop_scale_min == 0.85
    assert settings.brightness_jitter == 0.15
    assert settings.contrast_jitter == 0.15
    assert settings.rotation_degrees == 7.0
    assert not settings.resume_optimizer_state
    assert settings.output_dir.name.endswith("epoch141_augmented_kd")


def test_phase_slow_continuation_keeps_masks_trainable_at_lower_lr() -> None:
    settings = load_settings(
        EXPERIMENT
        / "configs"
        / "grocery10_replaced_continue_epoch141_augmented_kd_phase_slow.yaml"
    )
    assert settings.learning_rate == 1.0e-5
    assert settings.router_learning_rate == 2.0e-5
    assert settings.phase_learning_rate == 1.0e-6
    assert settings.phase_learning_rate < settings.learning_rate
    assert settings.output_dir.name.endswith("augmented_kd_phase_slow")


def test_phase_learning_rate_gets_an_independent_optimizer_group() -> None:
    class Core(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_phase = nn.Parameter(torch.zeros(2, 2))
            self.adapter = nn.Linear(2, 2)
            self.router = nn.Linear(2, 2)

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
        weight_decay=0.0,
    )
    optimizer, _ = _build_optimizer(replacement, readout, settings)
    groups = {
        group["group_name"]: group
        for group in optimizer.param_groups
    }
    assert groups["student_base"]["lr"] == 1.0e-5
    assert groups["routers"]["lr"] == 2.0e-5
    assert groups["optical_phases"]["lr"] == 1.0e-6
    phase_ids = {
        id(replacement.vision_surrogate.core.raw_phase),
        id(replacement.language_surrogate.core.raw_phase),
    }
    assert {id(value) for value in groups["optical_phases"]["params"]} == phase_ids


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


def test_continuation_config_adds_gallery_negatives_and_router_recovery() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "grocery10_continue100.yaml"
    )
    assert settings.epochs == 100
    assert settings.batch_size == 30
    assert settings.pk_images_per_sku == 3
    assert settings.lambda_kd == 3.0
    assert settings.lambda_gallery == 0.5
    assert settings.lambda_router_balance == 0.05
    assert settings.lambda_router_importance == 0.01
    assert settings.learning_rate == 0.0002
    assert settings.router_learning_rate == 0.001
    assert settings.router_temperature == 2.0
    assert not settings.resume_optimizer_state


def test_stable_epoch57_continuation_ends_at_epoch150() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "grocery10_continue_epoch57_stable.yaml"
    )
    assert settings.epochs == 93
    assert 57 + settings.epochs == 150
    assert settings.learning_rate == 0.00005
    assert settings.router_learning_rate == 0.0001
    assert settings.lambda_kd == 5.0
    assert settings.lambda_router_balance == 0.02
    assert settings.lambda_router_importance == 0.005


def test_fixed_gallery_continuation_ends_at_epoch150() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "grocery10_continue_epoch60_fixed_gallery.yaml"
    )
    assert settings.epochs == 90
    assert 60 + settings.epochs == 150
    assert settings.learning_rate == 0.00002
    assert settings.router_learning_rate == 0.00005
    assert settings.gallery_temperature == 0.15
    assert settings.gallery_prototype_stop_gradient


def test_epoch118_resume_ends_at_epoch150() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "grocery10_continue_epoch118_to150.yaml"
    )
    assert settings.epochs == 32
    assert 118 + settings.epochs == 150
    assert settings.gallery_prototype_stop_gradient


def test_student_has_one_expert_stage_plus_one_global_phase() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "grocery10.yaml")
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


def test_official_dataset_split_and_no_leakage(tmp_path: Path) -> None:
    root = tmp_path / "GroceryStoreDataset"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    base_settings = load_settings(EXPERIMENT / "configs" / "grocery10.yaml")
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
                "base_config": str(EXPERIMENT / "configs" / "grocery10.yaml"),
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
