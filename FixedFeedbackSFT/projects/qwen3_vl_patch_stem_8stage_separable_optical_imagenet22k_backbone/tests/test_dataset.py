from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone.dataset import (
    ClassFolderMMapDataset,
    GlobalAffineDistributedSampler,
    build_class_folder_index,
    verify_index,
)


def _source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "images"
    wnids = ["n0001", "n0002", "n0003"]
    for class_index, wnid in enumerate(wnids):
        directory = root / wnid
        directory.mkdir(parents=True)
        for image_index in range(5):
            Image.new("RGB", (8, 8), (class_index * 60, image_index * 20, 10)).save(
                directory / f"{image_index:02d}.jpg"
            )
    declaration = tmp_path / "declaration.json"
    declaration.write_text(
        json.dumps(
            {
                "format": "authorized-class-folder-source-v1",
                "dataset_id": "synthetic-test-only",
                "variant_id": "synthetic-test-only",
                "release_id": "unit-test",
                "split_id": "train",
                "expected_num_classes": 3,
                "expected_num_samples": 15,
                "class_wnids": wnids,
                "license_or_access_acknowledged": True,
            }
        ),
        encoding="utf-8",
    )
    return root, declaration


def test_disk_index_and_mmap_records(tmp_path: Path) -> None:
    root, declaration = _source(tmp_path)
    output = tmp_path / "index"
    manifest = build_class_folder_index(
        source_root=root,
        declaration_path=declaration,
        output_dir=output,
        strict_known_variant=False,
    )
    assert manifest["num_classes"] == 3
    assert manifest["num_samples"] == 15
    assert (output / "offsets.u64").stat().st_size == 16 * 8
    assert verify_index(output)["index_manifest_sha256"]
    dataset = ClassFolderMMapDataset(output, transform=None, verify_large_files=True)
    assert len(dataset) == 15
    assert dataset.record(0) == ("n0001/00.jpg", 0)
    assert dataset.record(14) == ("n0003/04.jpg", 2)
    assert dataset[6]["label"] == 1


def test_global_sampler_is_bijective_cross_class_and_ddp_sharded(tmp_path: Path) -> None:
    root, declaration = _source(tmp_path)
    output = tmp_path / "index"
    build_class_folder_index(
        source_root=root,
        declaration_path=declaration,
        output_dir=output,
        strict_known_variant=False,
    )
    dataset = ClassFolderMMapDataset(output, transform=None)
    rank0 = list(
        GlobalAffineDistributedSampler(
            dataset, rank=0, world_size=2, seed=2026, shuffle=True, pad_to_even=False
        )
    )
    rank1 = list(
        GlobalAffineDistributedSampler(
            dataset, rank=1, world_size=2, seed=2026, shuffle=True, pad_to_even=False
        )
    )
    assert sorted(rank0 + rank1) == list(range(len(dataset)))
    assert set(rank0).isdisjoint(rank1)
    labels = [dataset.record(index)[1] for index in rank0]
    assert len(set(labels[: min(6, len(labels))])) > 1
