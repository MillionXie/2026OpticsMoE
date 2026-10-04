"""Data contracts for the nine-class CRC domain-incremental experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ("tumor", "smooth_muscle", "cancer_stroma", "lymphocytes",
               "debris", "normal_mucosa", "adipose", "background", "mucus")
DOMAIN_NAMES = ("A_original", "B_stain_h", "C_stain_e", "D_scanner")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_domain(path, manifest_path):
    path, manifest_path = Path(path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("license") != "CC BY 4.0":
        raise ValueError("dataset license was not fixed to CC BY 4.0")
    if manifest.get("cache_sha256") != sha256(path):
        raise ValueError("data checksum mismatch")
    if tuple(manifest.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("class mapping mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = {f"{split}_{kind}" for split in ("train", "val", "test")
                    for kind in ("images", "labels", "ids")}
        if required.difference(archive.files):
            raise ValueError(f"missing arrays: {sorted(required.difference(archive.files))}")
        data = {name: archive[name].copy() for name in required}
    seen = set()
    for split in ("train", "val", "test"):
        images, labels, ids = (data[f"{split}_{kind}"] for kind in ("images", "labels", "ids"))
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images must be NHWC uint8 RGB")
        if set(labels.tolist()) != set(range(9)):
            raise ValueError(f"{split} does not contain all nine classes")
        if len(images) != len(labels) or len(labels) != len(ids) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"invalid {split} identities")
        if seen.intersection(ids.tolist()):
            raise ValueError("identity leakage across splits")
        seen.update(ids.tolist())
    return data, manifest


def balanced_subset(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(9):
        indices = np.flatnonzero(labels == label)
        if len(indices) < per_class:
            raise ValueError(f"class {label} has {len(indices)}, requested {per_class}")
        chosen.extend(rng.permutation(indices)[:per_class])
    return np.asarray(chosen, dtype=np.int64)

