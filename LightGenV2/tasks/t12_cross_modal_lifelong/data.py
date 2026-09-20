"""Load the three audited common-field datasets and prepare CLEVR fields."""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t09_multimodal_matching.model import TextEncoder
from LightGenV2.tasks.t09_multimodal_matching.prepare import tokens


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest.get("files", manifest)
    for name, expected in files.items():
        path = root / name
        if path.is_file() and len(expected) == 64:
            assert sha256(path) == expected, name
    return sha256(root / "manifest.json")


def load_common(root, split):
    root = Path(root)
    d = np.load(root / f"{split}.npz")
    rows_path = root / f"{split}_records.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else [{} for _ in d["labels"]]
    return torch.from_numpy(d["fields"]), np.asarray(d["labels"], dtype=np.int64), rows


def normalize_power(x, power):
    return x * (power / x.square().sum((-2, -1), keepdim=True).clamp_min(1e-20)).sqrt()


def encode_clevr(images, word_grid):
    # Pack all RGB channels into the left half and text into the right half.
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("CLEVR images must be NHWC RGB")
    rgb = images.permute(0, 3, 1, 2).float() / 255.0
    rgb = F.interpolate(rgb, (112, 56), mode="bilinear", align_corners=False)
    blank = torch.zeros_like(rgb[:, 0])
    visual = torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                        torch.cat((rgb[:, 2], blank), -1)), -2)
    text = F.interpolate(word_grid[:, None], (224, 112), mode="nearest")[:, 0]
    return torch.cat((normalize_power(visual, 0.5), normalize_power(text, 0.5)), -1)


def prepare_clevr(source, out):
    """Materialize the existing fixed text/RGB encoding into the common format.

    The historical CLEVR package has train/val only.  We deterministically split
    its image-disjoint validation set by image id into validation and test halves.
    No model score participates in that split.
    """
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source_manifest_sha = verify_manifest(source)
    vocab = json.loads((source / "vocab.json").read_text())
    encoder = TextEncoder(len(vocab), "fixed").eval()

    def read(split):
        images = torch.from_numpy(np.load(source / f"{split}_images.npz")["images"])
        rows = json.loads((source / f"{split}_questions.json").read_text())
        ids = torch.zeros(len(rows), 32, dtype=torch.long)
        for i, row in enumerate(rows):
            words = tokens(row["question"])
            ids[i, :len(words)] = torch.tensor([vocab.get(w, 1) for w in words])
        fields = []
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                index = torch.tensor([r["image_local"] for r in rows[i:i+128]])
                fields.append(encode_clevr(images[index], encoder(ids[i:i+128])))
        return torch.cat(fields).numpy().astype(np.float16), np.array([r["label"] for r in rows]), rows

    train = read("train")
    held = read("val")
    image_ids = sorted({str(r["image_id"]) for r in held[2]}, key=lambda s: hashlib.sha256(("17:"+s).encode()).hexdigest())
    val_ids = set(image_ids[::2])
    masks = {"val": np.array([str(r["image_id"]) in val_ids for r in held[2]])}
    masks["test"] = ~masks["val"]
    splits = {"train": train}
    for split, mask in masks.items():
        splits[split] = (held[0][mask], held[1][mask], [r for r, keep in zip(held[2], mask) if keep])
    for split, (fields, labels, rows) in splits.items():
        np.savez_compressed(out / f"{split}.npz", fields=fields, labels=labels.astype(np.int64))
        (out / f"{split}_records.json").write_text(json.dumps(rows, indent=2) + "\n")
    protocol = {"task": "clevr", "classes": 2,
                "scope": "existing CC-BY-4.0 attribute query package; source validation images split 50/50 into model-independent val/test",
                "input": "RGB channels packed as [R,G;B,zero] in left half; fixed word-position one-hot in right half; power 0.5 each",
                "source_manifest_sha256": source_manifest_sha,
                "split_seed": 17}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare_clevr(a.source, a.out)


if __name__ == "__main__":
    main()
