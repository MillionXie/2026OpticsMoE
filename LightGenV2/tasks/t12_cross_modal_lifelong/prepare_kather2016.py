"""Convert every Kather2016 image into the common optical field format."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CLASSES = ["tumor", "stroma", "complex_stroma", "lymphocytes",
           "debris", "mucosa", "adipose", "background"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_rgb(images):
    """Preserve RGB and mean intensity in four equal-area optical tiles."""
    x = torch.as_tensor(images, dtype=torch.float32)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB, got {tuple(x.shape)}")
    if x.max() > 1:
        x = x / 255.0
    x = F.interpolate(x.permute(0, 3, 1, 2), (112, 112), mode="bilinear",
                      align_corners=False, antialias=True)
    mean = x.mean(1)
    field = torch.cat((torch.cat((x[:, 0], x[:, 1]), -1),
                       torch.cat((x[:, 2], mean), -1)), -2)
    power = field.square().sum((-2, -1), keepdim=True).clamp_min(1e-20)
    return (field / power.sqrt()).numpy().astype(np.float16)


def prepare(source, out):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source_npz = source / "kather2016_fixed_split.npz"
    source_manifest = source / "data_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf8"))
    if manifest.get("raw_images") != 5000 or manifest.get("unique_images") != 5000:
        raise ValueError("Kather2016 source must contain all 5,000 unique images")
    data = np.load(source_npz)
    total = 0
    for split in ("train", "val", "test"):
        images = data[f"{split}_images"]
        labels = np.asarray(data[f"{split}_labels"], dtype=np.int64)
        ids = [str(x) for x in data[f"{split}_ids"]]
        fields = []
        for start in range(0, len(labels), 128):
            fields.append(encode_rgb(images[start:start + 128]))
        np.savez_compressed(out / f"{split}.npz", fields=np.concatenate(fields), labels=labels)
        rows = [{"image_id": image_id, "class": CLASSES[int(label)]}
                for image_id, label in zip(ids, labels)]
        (out / f"{split}_records.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf8")
        total += len(labels)
    if total != 5000:
        raise ValueError(f"prepared {total} images, expected 5000")
    protocol = {
        "task": "kather2016", "classes": 8,
        "dataset": "Kather texture 2016 image tiles",
        "all_original_samples": True, "source_images": 5000,
        "split_sizes": {s: int(len(data[f"{s}_labels"])) for s in ("train", "val", "test")},
        "scope": "all 5,000 original images; fixed stratified image-level split",
        "input": "RGB and RGB mean tiled [R,G;B,mean] into 224x224; unit power",
        "class_names": CLASSES,
        "license": "CC BY 4.0", "dataset_doi": "10.5281/zenodo.53169",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "source_cache_sha256": sha256(source_npz),
        "limitations": "The public package has no auditable patient/slide identifiers; the split is image-level."
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf8")
    files = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.out)
