from __future__ import annotations

from pathlib import Path

from PIL import Image

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa import datasets


def _write_split(root: Path, split: str, count: int, classes: int) -> None:
    image_dir = root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    image = image_dir / "sample.png"
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(image)
    lines = [f"images/{split}/sample.png {index % classes}" for index in range(count)]
    (root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_standard_split_loader_and_qwen_tensor(tmp_path: Path, monkeypatch) -> None:
    spec = datasets.TaskSpec("tiny", "tiny", "natural", 2, 20)
    monkeypatch.setitem(datasets.TASK_SPECS, "tiny", spec)
    root = tmp_path / "tiny"
    for split, count in {
        "train800": 800,
        "val200": 200,
        "train800val200": 1_000,
        "test": 20,
    }.items():
        _write_split(root, split, count, classes=2)
    dataset = datasets.Vtab1kClassificationDataset(tmp_path, "tiny", "train800")
    sample = dataset[0]
    assert len(dataset) == 800
    assert tuple(sample["image"].shape) == (3, 224, 224)
    assert sample["label"].item() == 0
    assert len(dataset.manifest_sha256()) == 64


def test_structured_task_class_counts_are_explicit() -> None:
    assert datasets.TASK_SPECS["dsprites_orientation"].num_classes == 16
    assert datasets.TASK_SPECS["smallnorb_azimuth"].num_classes == 18
