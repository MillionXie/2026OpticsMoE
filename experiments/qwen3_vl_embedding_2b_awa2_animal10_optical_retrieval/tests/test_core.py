from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.prepare_awa2_retrieval_subset import (
    AwA2Sample,
    prepare_awa2_subset,
)
from experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.cache_teacher_embeddings import (
    _derive_subset_cache,
    cache_identity,
)
from experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.retrieval_metrics import (
    evaluate_embeddings,
)
from experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
)
from experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.train_optical_retrieval import (
    PKBatchSampler,
    supervised_contrastive_loss,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def _fake_awa2(root: Path, names: tuple[str, ...], count: int = 12) -> None:
    image_root = root / "Animals_with_Attributes2" / "JPEGImages"
    for class_index, name in enumerate(names):
        directory = image_root / name
        directory.mkdir(parents=True, exist_ok=True)
        for image_index in range(count):
            value = (class_index * 23 + image_index) % 255
            Image.new("RGB", (20, 18), (value, 255 - value, 64)).save(
                directory / f"{name}_{image_index:04d}.jpg"
            )


def test_release_configs_resolve_inside_experiment() -> None:
    stage1 = load_settings(EXPERIMENT / "configs" / "awa2_50class_pretrain.yaml")
    stage2 = load_settings(EXPERIMENT / "configs" / "awa2_10class_finetune.yaml")
    assert stage1.use_all_classes is True
    assert stage1.epochs == 30
    assert stage2.use_all_classes is False
    assert len(stage2.selected_classes) == 10
    assert stage2.epochs == 20
    assert stage1.output_dir.parent == EXPERIMENT / "runs"
    assert stage2.output_dir.parent == EXPERIMENT / "runs"
    assert stage2.oeo_preserve_response_amplitude is True
    assert stage2.num_experts == 4 and stage2.top_k == 2


def test_fixed_split_is_disjoint_and_persistent(tmp_path: Path) -> None:
    base = load_settings(EXPERIMENT / "configs" / "smoke_10class.yaml")
    names = base.selected_classes
    root = tmp_path / "AwA2"
    _fake_awa2(root, names)
    settings = replace(
        base,
        dataset_root=root,
        output_dir=tmp_path / "run",
        download=False,
        train_limit_per_class=None,
        test_limit_per_class=None,
        gallery_images_per_class=2,
    )
    first = prepare_awa2_subset(settings, persist=True)
    second = prepare_awa2_subset(settings, persist=False)
    assert first.manifest_digest == second.manifest_digest
    assert len(first.gallery_samples) == 2 * len(names)
    paths = [
        {sample.image_path for sample in values}
        for values in (first.train_samples, first.test_samples, first.gallery_samples)
    ]
    assert not paths[0] & paths[1]
    assert not paths[0] & paths[2]
    assert not paths[1] & paths[2]
    assert settings.subset_manifest_path.is_file()
    assert (settings.output_dir / "dataset.json").is_file()


def test_class_subset_keeps_same_image_roles(tmp_path: Path) -> None:
    base = load_settings(EXPERIMENT / "configs" / "smoke_10class.yaml")
    names = base.selected_classes
    root = tmp_path / "AwA2"
    _fake_awa2(root, names)
    all_target = replace(
        base,
        dataset_root=root,
        output_dir=tmp_path / "all",
        download=False,
        train_limit_per_class=None,
        test_limit_per_class=None,
    )
    subset = replace(
        all_target,
        selected_classes=names[:3],
        output_dir=tmp_path / "subset",
        pk_classes_per_batch=3,
        pk_images_per_class=2,
        batch_size=6,
    )
    full_bundle = prepare_awa2_subset(all_target, persist=False)
    subset_bundle = prepare_awa2_subset(subset, persist=False)
    for split_name in ("train_samples", "test_samples", "gallery_samples"):
        full = {
            (sample.class_name, sample.image_path)
            for sample in getattr(full_bundle, split_name)
            if sample.class_name in names[:3]
        }
        selected = {
            (sample.class_name, sample.image_path)
            for sample in getattr(subset_bundle, split_name)
        }
        assert full == selected


def test_pk_sampler_and_supervised_contrastive_have_positives(tmp_path: Path) -> None:
    samples = []
    for class_index in range(4):
        for image_index in range(3):
            samples.append(
                AwA2Sample(
                    f"{class_index}:{image_index}",
                    tmp_path / f"{class_index}_{image_index}.jpg",
                    class_index + 1,
                    f"class{class_index}",
                    class_index,
                    "train",
                    "deterministic_train",
                    False,
                )
            )
    sampler = PKBatchSampler(samples, 4, 2, seed=42, steps_per_epoch=2)
    batches = list(iter(sampler))
    assert len(batches) == 2
    for batch in batches:
        labels = torch.tensor([samples[index].class_index for index in batch])
        assert all(int(labels.eq(value).sum()) == 2 for value in labels.unique())
        embedding = torch.nn.functional.normalize(torch.randn(len(batch), 64), dim=-1)
        assert torch.isfinite(supervised_contrastive_loss(embedding, labels, 0.07))


def test_retrieval_metrics_are_class_level(tmp_path: Path) -> None:
    names = ("zebra", "dolphin", "horse")
    gallery = [
        AwA2Sample(f"g{i}", tmp_path / f"g{i}.jpg", i + 1, name, i, "gallery", "x", True)
        for i, name in enumerate(names)
    ]
    query = [
        AwA2Sample(f"q{i}", tmp_path / f"q{i}.jpg", i + 1, name, i, "test", "x", False)
        for i, name in enumerate(names)
    ]
    values = torch.eye(3)
    result = evaluate_embeddings(
        values, query, values, gallery, names, "mean_prototype", system_name="test"
    )
    assert result.metrics["top1_retrieval_accuracy"] == pytest.approx(1.0)
    assert result.metrics["top3_retrieval_accuracy"] == pytest.approx(1.0)
    assert result.metrics["mrr"] == pytest.approx(1.0)


def test_moe4_optical_core_forward_backward_is_finite() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "smoke_10class.yaml")
    core = HomogeneousMoEOpticalCore(hidden_size=32, max_tokens=224, settings=settings)
    hidden = torch.randn(6, 32, requires_grad=True)
    input_fields = core.encode_groups([hidden])
    field, routing = core.begin(input_fields)
    assert field.shape == (1, 518, 518)
    assert field.dtype == torch.complex64
    assert int(routing["selected_mask"].sum()) == 2
    field = core.run_stage(0, field, routing)
    output = core.read_hidden(field, [6], torch.float32, final=True)
    assert output.shape == (6, 32)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    phase_gradients = [
        parameter.grad
        for name, parameter in core.named_parameters()
        if "raw_phase" in name
    ]
    assert phase_gradients and all(value is not None for value in phase_gradients)
    assert all(torch.isfinite(value).all() for value in phase_gradients if value is not None)


def test_target_teacher_cache_is_sliced_without_qwen_forward(tmp_path: Path) -> None:
    base = load_settings(EXPERIMENT / "configs" / "smoke_10class.yaml")
    names = base.selected_classes
    root = tmp_path / "AwA2"
    _fake_awa2(root, names, count=12)
    settings = replace(
        base,
        dataset_root=root,
        output_dir=tmp_path / "target_run",
        download=False,
        train_limit_per_class=None,
        test_limit_per_class=None,
    )
    bundle = prepare_awa2_subset(settings, persist=False)
    samples = bundle.all_samples()
    embeddings = torch.nn.functional.normalize(torch.randn(len(samples), 64), dim=-1).half()
    source_path = tmp_path / "all50_cache.pt"
    torch.save(
        {
            "metadata": cache_identity(bundle, settings),
            "records": [sample.manifest_record() for sample in samples],
            "teacher_embeddings": embeddings,
        },
        source_path,
    )
    destination = settings.teacher_cache_path
    result = _derive_subset_cache(source_path, destination, bundle, settings)
    assert result == destination
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    assert payload["metadata"]["derived_without_teacher_forward"] is True
    assert payload["teacher_embeddings"].shape == (len(samples), 64)
