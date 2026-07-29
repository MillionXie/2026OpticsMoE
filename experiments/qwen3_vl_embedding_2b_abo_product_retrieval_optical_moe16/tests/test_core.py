from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16.datasets import (
    CatalogItem,
    ImageRecord,
    assert_no_image_leakage,
    build_fixed_manifest,
    load_product_image,
    split_stage2_images,
)
from experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16.evaluation import (
    evaluate_retrieval,
)
from experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16.losses import (
    cosine_distillation_loss,
    supervised_contrastive_loss,
)
from experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16.modeling import (
    DetectorTokenProjection,
    TrainingIdentityHead,
)
from experiments.qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16.training import (
    PKBatchSampler,
)


def test_five_view_split_is_three_one_one_and_stable() -> None:
    images = [f"image-{index}" for index in range(5)]
    first = split_stage2_images("item", images, 42)
    second = split_stage2_images("item", list(reversed(images)), 42)
    assert first == second
    assert list(first.values()).count("train") == 3
    assert list(first.values()).count("gallery") == 1
    assert list(first.values()).count("query") == 1


def test_main_catalog_image_can_be_fixed_as_gallery() -> None:
    images = [f"image-{index}" for index in range(5)]
    split = split_stage2_images(
        "item", images, 42, preferred_gallery_ids=["image-3"]
    )
    assert split["image-3"] == "gallery"


def test_product_preprocessing_letterboxes_without_stretching(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), (10, 20, 30)).save(path)
    output = load_product_image(path, 224)
    assert output.size == (224, 224)
    # The 2:1 source occupies 224x112 and is vertically centered on white.
    assert output.getpixel((112, 0)) == (255, 255, 255)
    assert output.getpixel((112, 112)) == (10, 20, 30)


def test_fixed_manifest_excludes_gallery_and_query_from_both_training_stages() -> None:
    items = []
    for item_index in range(5):
        images = tuple(
            ImageRecord(
                image_id=f"{item_index}-{view}",
                relative_path=f"{item_index}/{view}.jpg",
                width=512,
                height=512,
                is_main=view == 0,
            )
            for view in range(5)
        )
        items.append(CatalogItem(f"item-{item_index}", "TYPE", images))
    settings = SimpleNamespace(
        stage2_min_images_per_item=4,
        stage2_item_count=3,
        preferred_product_types=(),
        stage2_product_type_count=1,
        quality_candidate_multiplier=1,
        quality_scan_enabled=False,
        dataset_root=Path("."),
        stage2_max_images_per_item=5,
        random_seed=42,
        stage2_train_fraction=0.6,
        stage2_gallery_fraction=0.2,
        stage2_query_fraction=0.2,
        stage1_min_images_per_item=2,
        stage1_max_images_per_item=5,
        stage1_item_count=5,
        stage1_target_image_count=15,
    )
    rows, metadata = build_fixed_manifest(items, settings)
    assert_no_image_leakage(rows)
    held_out = {
        row["image_id"]
        for row in rows
        if row["stage2_split"] in {"gallery", "query"}
    }
    training = {
        row["image_id"]
        for row in rows
        if row["stage1_train"] or row["stage2_split"] == "train"
    }
    assert not (held_out & training)
    assert metadata["stage2_items"] == 3


def test_leakage_guard_rejects_held_out_image_in_stage1() -> None:
    rows = [
        {
            "image_id": "same",
            "item_id": "item",
            "stage1_train": 1,
            "stage2_split": "query",
        },
        {
            "image_id": "gallery",
            "item_id": "item",
            "stage1_train": 0,
            "stage2_split": "gallery",
        },
        {
            "image_id": "train",
            "item_id": "item",
            "stage1_train": 0,
            "stage2_split": "train",
        },
    ]
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_image_leakage(rows)


def test_detector_projection_outputs_signed_224d_unit_embeddings() -> None:
    module = DetectorTokenProjection(224)
    detector = torch.rand(2, 224, 224, requires_grad=True)
    output = module(detector, [196, 200])
    assert output.shape == (2, 224)
    torch.testing.assert_close(
        output.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5
    )
    output.square().sum().backward()
    assert module.projection.weight.grad is not None


def test_losses_use_same_item_positives_and_normalized_teacher() -> None:
    embeddings = torch.randn(6, 224, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    supcon = supervised_contrastive_loss(embeddings, labels, 0.07)
    teacher = torch.nn.functional.normalize(torch.randn(6, 224), dim=-1)
    kd = cosine_distillation_loss(
        torch.nn.functional.normalize(embeddings, dim=-1), teacher
    )
    (supcon + kd).backward()
    assert torch.isfinite(supcon)
    assert embeddings.grad is not None


def test_pk_sampler_preserves_p_by_k_structure() -> None:
    samples = [
        SimpleNamespace(item_index=item, item_id=str(item))
        for item in range(4)
        for _ in range(3)
    ]
    sampler = PKBatchSampler(samples, p=2, k=2, seed=42)
    for batch in sampler:
        labels = [samples[index].item_index for index in batch]
        counts = {label: labels.count(label) for label in set(labels)}
        assert sorted(counts.values()) == [2, 2]


def test_item_level_retrieval_metrics() -> None:
    gallery = torch.eye(4, dtype=torch.float32)
    query = gallery.clone()
    item_ids = ["a", "b", "c", "d"]
    metrics, rows = evaluate_retrieval(
        query,
        item_ids,
        gallery,
        item_ids,
        aggregation="mean_prototype",
    )
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mean_average_precision"] == 1.0
    assert all(row["rank"] == 1 for row in rows)


def test_stage2_identity_head_is_explicitly_training_only() -> None:
    head = TrainingIdentityHead(224, 500)
    output = head(torch.randn(3, 224))
    assert output.shape == (3, 500)
    assert set(head.state_dict()) == {
        "classifier.weight",
        "classifier.bias",
    }
