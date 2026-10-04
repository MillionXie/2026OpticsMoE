"""Prepare four deterministic stain/scanner domains from CC BY 4.0 CRC tiles."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
from PIL import Image

from .crc9_data import CLASS_NAMES, DOMAIN_NAMES, sha256


OD_GAINS = {
    "A_original": (1.00, 1.00, 1.00),
    "B_stain_h": (1.15, .90, 1.05),
    "C_stain_e": (.90, 1.12, 1.05),
    "D_scanner": (1.08, 1.04, .88),
}


def read_arrow(path):
    with open(path, "rb") as handle:
        table = ipc.open_stream(handle).read_all()
    labels = np.asarray(table["label"].to_pylist(), dtype=np.int64)
    names = table["label_name"].to_pylist()
    mapping = {int(label): name for label, name in zip(labels, names)}
    if tuple(mapping[i] for i in range(9)) != CLASS_NAMES:
        raise ValueError(f"unexpected class map: {mapping}")
    licenses = {str(value).replace("-", " ").upper() for value in table["license"].to_pylist()}
    if licenses != {"CC BY 4.0"}:
        raise ValueError(f"unexpected licenses: {licenses}")
    images = []
    for record in table["image"].to_pylist():
        with Image.open(io.BytesIO(record["bytes"])) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(images), labels, np.asarray(table["image_id"].to_pylist(), dtype="U64")


def transform(images, gains):
    if gains == (1., 1., 1.):
        return images.copy()
    value = images.astype(np.float32)
    optical_density = -np.log((value + 1.) / 256.)
    optical_density *= np.asarray(gains, np.float32)[None, None, None, :]
    return np.clip(np.exp(-optical_density) * 256. - 1., 0, 255).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-arrow", type=Path, required=True)
    parser.add_argument("--validation-arrow", type=Path, required=True)
    parser.add_argument("--test-arrow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    raw = {"train": read_arrow(args.train_arrow),
           "val": read_arrow(args.validation_arrow),
           "test": read_arrow(args.test_arrow)}
    source = {name: {"path": str(path), "sha256": sha256(path)} for name, path in
              (("train", args.train_arrow), ("validation", args.validation_arrow), ("test", args.test_arrow))}
    for domain in DOMAIN_NAMES:
        payload = {}
        for split, (images, labels, ids) in raw.items():
            payload[f"{split}_images"] = transform(images, OD_GAINS[domain])
            payload[f"{split}_labels"] = labels
            payload[f"{split}_ids"] = ids
        target = args.out / f"{domain}.npz"
        np.savez_compressed(target, **payload)
        manifest = {
            "dataset": "Kather et al. 2018 colorectal histology, controlled domain shifts",
            "domain": domain, "domain_transform": "optical-density channel gains",
            "optical_density_gains": list(OD_GAINS[domain]),
            "license": "CC BY 4.0", "class_names": list(CLASS_NAMES),
            "source": source, "cache_sha256": sha256(target),
            "sizes": {split: len(values[1]) for split, values in raw.items()},
            "paired_across_domains": True,
            "note": "Domains are deterministic controlled color shifts of the same split identities; they are not independent clinical cohorts."
        }
        (args.out / f"{domain}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
