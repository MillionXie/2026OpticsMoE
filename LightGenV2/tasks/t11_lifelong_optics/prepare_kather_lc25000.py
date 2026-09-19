"""Build licensed binary caches for Kather2016 -> LC25000 continual learning."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .data import sha


LC25000_MD5 = "1b1325f690bc51fd76bb8c4958c03b06"


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path, dataset, cache, source, split_protocol, limitations):
    path.write_text(json.dumps({
        "dataset": dataset,
        "license": "CC BY 4.0",
        "source": source,
        "cache_sha256": sha(cache),
        "split_protocol": split_protocol,
        "limitations": limitations,
        "label_map": {"0": "tumor", "1": "non_tumor"},
    }, indent=2))


def prepare_kather(source, source_manifest, output):
    metadata = json.loads(source_manifest.read_text())
    if metadata.get("license") != "CC BY 4.0" or metadata.get("cache_sha256") != sha(source):
        raise ValueError("Kather source license/hash check failed")
    with np.load(source, allow_pickle=False) as z:
        arrays = {}
        for split in ("train", "val"):
            arrays[f"{split}_images"] = z[f"{split}_images"].copy()
            arrays[f"{split}_labels"] = (z[f"{split}_labels"] != 0).astype(np.int64)
            arrays[f"{split}_ids"] = np.asarray(["kather2016:" + str(v) for v in z[f"{split}_ids"]])
    np.savez_compressed(output, **arrays)
    write_manifest(output.with_name("kather2016_binary_manifest.json"), "Kather2016 binary", output,
                   "https://zenodo.org/records/53169", "Existing fixed train/validation image split; test arrays not read.",
                   "Image-level split; not claimed patient-independent. Tumor is original class 0; all other tissues are non-tumor.")


def prepare_lc25000(source, output, seed):
    if md5(source) != LC25000_MD5:
        raise ValueError("LC25000 archive MD5 mismatch")
    rng = np.random.default_rng(seed); classes = {"colon_aca": 0, "colon_n": 1}; records = {}
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        for class_name, label in classes.items():
            candidates = sorted(name for name in names if f"/{class_name}/" in name and name.lower().endswith((".jpg", ".jpeg", ".png")))
            if len(candidates) != 5000:
                raise ValueError(f"Expected 5000 {class_name} images, found {len(candidates)}")
            order = rng.permutation(len(candidates)); records[class_name] = (label, candidates, order)
        arrays = {"train_images": [], "train_labels": [], "train_ids": [], "val_images": [], "val_labels": [], "val_ids": []}
        for class_name, (label, candidates, order) in records.items():
            for split, selected in (("train", order[:4000]), ("val", order[4000:])):
                for index in selected:
                    name = candidates[index]
                    with Image.open(io.BytesIO(archive.read(name))) as image:
                        image = image.convert("RGB").resize((150, 150), Image.Resampling.LANCZOS)
                        arrays[f"{split}_images"].append(np.asarray(image, dtype=np.uint8))
                    arrays[f"{split}_labels"].append(label); arrays[f"{split}_ids"].append("lc25000:" + name)
    packed = {}
    for key, values in arrays.items():
        packed[key] = np.stack(values) if key.endswith("images") else np.asarray(values, dtype=np.int64 if key.endswith("labels") else str)
    np.savez_compressed(output, **packed)
    write_manifest(output.with_name("lc25000_colon_binary_manifest.json"), "LC25000 colon binary", output,
                   "https://zenodo.org/records/14998042", "Seeded stratified 4000/1000 image-level split per class.",
                   "LC25000 contains augmented derivatives of a smaller source collection; augmentation-family and patient identifiers are unavailable, so this split is not claimed patient-independent.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kather", type=Path, required=True)
    parser.add_argument("--kather-manifest", type=Path, required=True)
    parser.add_argument("--lc25000-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    prepare_kather(args.kather, args.kather_manifest, args.out / "kather2016_binary.npz")
    prepare_lc25000(args.lc25000_zip, args.out / "lc25000_colon_binary.npz", args.seed)


if __name__ == "__main__":
    main()
