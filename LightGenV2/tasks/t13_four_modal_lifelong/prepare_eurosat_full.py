"""Decode every usable paired EuroSAT RGB/SAR scene in the audited split."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
PREPARE = HERE.parents[1] / "demo_check/pure_optical/prepare.py"
spec = importlib.util.spec_from_file_location("eurosat_pair_decoder", PREPARE)
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(rows, archives, maps, split):
    images, labels, domains, ids = [], [], [], []
    for index, row in enumerate(rows):
        rgb, sar = decoder.decode_pair(row["pair_id"], archives, maps)
        for domain, image in enumerate((rgb, sar)):
            images.append(image)
            labels.append(row["label"])
            domains.append(domain)
            ids.append(row["pair_id"] + ":" + str(domain))
        if (index + 1) % 500 == 0:
            print(json.dumps({"split": split, "pairs": index + 1,
                              "total_pairs": len(rows)}), flush=True)
    return {split + "_images": np.stack(images),
            split + "_labels": np.asarray(labels, dtype=np.int64),
            split + "_domains": np.asarray(domains, dtype=np.int8),
            split + "_ids": np.asarray(ids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Directory containing SPLIT.json and the three archives")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    split_path = a.root / "SPLIT.json"
    if decoder.digest(split_path, "sha256") != "cfe8373dd33cc0fe64f083b9ca32377e767c21b078c3f1d91f2dacecc25cb776":
        raise ValueError("unexpected EuroSAT split")
    records = json.loads(split_path.read_text())["records"]
    rows = {name: [r for r in records if r["domain"] == "A" and r["split"] == name]
            for name in ("train", "validation", "test")}
    # Each source archive contains 27,000 files.  The immutable audited split
    # contains the 26,892 pairs that survived geospatial pairing/QC; do not
    # silently add the 108 files that have no record in that protocol.
    source_pairs = sum(map(len, rows.values()))
    if source_pairs != 26892:
        raise ValueError(f"unexpected usable pair count: {source_pairs}")
    groups = {name: {r["spatial_group"] for r in value} for name, value in rows.items()}
    if any(groups[a] & groups[b] for i, a in enumerate(groups)
           for b in list(groups)[i + 1:]):
        raise ValueError("spatial groups overlap between splits")

    archives, maps, source_hashes = {}, {}, {}
    for name, (url, filename, size, algorithm, expected) in decoder.SOURCES.items():
        path = a.root / "archives" / filename
        if path.stat().st_size != size or decoder.digest(path, algorithm) != expected:
            raise ValueError(f"archive verification failed: {name}")
        source_hashes[name] = {"url": url, "sha256": sha(path)}
        archives[name] = zipfile.ZipFile(path)
        suffix = ".jpg" if name == "rgb" else ".tif"
        maps[name] = {Path(x.filename).stem: x for x in archives[name].infolist()
                      if Path(x.filename).suffix.lower() == suffix
                      and not x.filename.startswith("__MACOSX/")}
        if len(maps[name]) != 27000:
            raise ValueError(f"archive member count failed: {name}")

    trainval = {}
    trainval.update(pack(rows["train"], archives, maps, "train"))
    trainval.update(pack(rows["validation"], archives, maps, "validation"))
    np.savez_compressed(a.out / "trainval.npz", **trainval)
    np.savez_compressed(a.out / "test.npz", **pack(rows["test"], archives, maps, "test"))
    protocol = {
        "task": "eurosat", "classes": 10, "storage": "eurosat_rgb_sar_v1",
        "all_original_samples": True,
        "source_archive_images_per_modality": 27000,
        "source_pairs": source_pairs,
        "excluded_by_published_split_or_pairing_qc": 27000 - source_pairs,
        "pair_counts": {name: len(value) for name, value in rows.items()},
        "image_counts": {name: 2 * len(value) for name, value in rows.items()},
        "split": "original audited spatial-group-disjoint split; no subsampling",
        "trainval_npz": str((a.out / "trainval.npz").resolve()),
        "holdout_npz": str((a.out / "test.npz").resolve()),
        "modalities": ["RGB", "Sentinel-1 SAR"],
        "objective": "10-class land-cover classification",
        "license": "MIT",
        "source_archives": source_hashes,
    }
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}
    (a.out / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
