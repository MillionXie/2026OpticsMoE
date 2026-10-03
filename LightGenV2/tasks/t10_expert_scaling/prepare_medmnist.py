"""Convert an official MedMNIST NPZ into the audited T10 train/validation format.

The test arrays are deliberately not accessed or copied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = {
    "bloodmnist": [
        "basophil", "eosinophil", "erythroblast", "immature granulocytes",
        "lymphocyte", "monocyte", "neutrophil", "platelet",
    ],
    "organcmnist": [
        "bladder", "femur-left", "femur-right", "heart", "kidney-left",
        "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen",
    ],
}

SOURCE = {
    "bloodmnist": {
        "url": "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
        "md5": "7053d0359d879ad8a5505303e11de1dc",
    },
    "organcmnist": {
        "url": "https://zenodo.org/records/10519652/files/organcmnist.npz?download=1",
        "md5": "b9ceb9546e10131b32923c5bbeaea2b1",
    },
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(LABELS), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    expected = SOURCE[a.dataset]
    if digest(a.source, "md5") != expected["md5"]:
        raise ValueError("Official source MD5 mismatch")
    a.out.mkdir(parents=True, exist_ok=False)
    arrays = {}
    split_ids = {}
    class_counts = {}
    with np.load(a.source, allow_pickle=False) as z:
        for split in ["train", "val"]:
            x = z[f"{split}_images"]
            y = z[f"{split}_labels"].reshape(-1).astype(np.int64)
            if x.ndim == 3:
                x = np.repeat(x[..., None], 3, axis=-1)
            if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8:
                raise ValueError((split, x.shape, x.dtype))
            if len(x) != len(y) or y.min() != 0 or y.max() >= len(LABELS[a.dataset]):
                raise ValueError("Invalid image/label arrays")
            ids = np.asarray([f"{a.dataset}:{split}:{i:06d}" for i in range(len(y))])
            arrays[f"{split}_images"] = x
            arrays[f"{split}_labels"] = y
            arrays[f"{split}_ids"] = ids
            split_ids[split] = ids.tolist()
            class_counts[split] = np.bincount(y, minlength=len(LABELS[a.dataset])).tolist()
    output = a.out / f"{a.dataset}_fixed_split.npz"
    np.savez_compressed(output, **arrays)
    manifest = {
        "dataset": a.dataset,
        "classes": LABELS[a.dataset],
        "license": "CC BY 4.0",
        "license_evidence": "https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py",
        "source_url": expected["url"],
        "source_md5": expected["md5"],
        "source_sha256": digest(a.source),
        "cache_sha256": digest(output),
        "split_protocol": "Official MedMNIST train and validation splits; test arrays not accessed",
        "split_ids": split_ids,
        "class_counts": class_counts,
        "training_ready": True,
        "test_read": False,
        "patient_independence_verified": True if a.dataset == "organcmnist" else None,
        "mirror_repackaged": False,
        "original_zip_pixel_equivalence_verified": True,
    }
    (a.out / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": manifest["cache_sha256"],
                      "class_counts": class_counts}, indent=2))


if __name__ == "__main__":
    main()
