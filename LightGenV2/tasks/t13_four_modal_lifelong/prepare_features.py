"""Freeze the already validated per-modality frontends into compact features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from LightGenV2.tasks.t09_multimodal_matching.prepare import tokens
from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
from LightGenV2.demo_check.shared_frontend.model import SharedFrontend
from .physical_frontend import PhysicalEncoder
from .data import sha256


def batches(x, n=256):
    for i in range(0, len(x), n):
        yield x[i:i+n]


@torch.no_grad()
def vision_features(model, images, device):
    result = []
    model.eval().to(device)
    for x in batches(images):
        value = torch.as_tensor(np.array(x, copy=True), device=device)
        result.append(model(value)[0].cpu().numpy().astype(np.float16))
    return np.concatenate(result)


@torch.no_grad()
def eurosat_features(checkpoint, images, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model = SharedFrontend(state).eval().to(device)
    result = []
    for x in batches(images):
        value = torch.as_tensor(np.array(x, copy=True), device=device)
        result.append(model(value).cpu().numpy().astype(np.float16))
    return np.concatenate(result)


def token_array(rows, vocab):
    ids = np.zeros((len(rows), 32), dtype=np.uint8)
    for i, row in enumerate(rows):
        words = tokens(row["question"])
        ids[i, :len(words)] = [vocab.get(word, 1) for word in words]
    return ids


def write_split(out, split, features, labels, rows, token_ids=None):
    np.save(out / f"{split}_features.npy", np.asarray(features, dtype=np.float16))
    np.save(out / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
    if token_ids is not None:
        np.save(out / f"{split}_token_ids.npy", np.asarray(token_ids, dtype=np.uint8))
    (out / f"{split}_records.json").write_text(json.dumps(rows, indent=2) + "\n")


def prepare_eurosat(source, checkpoint, out, device):
    protocol = json.loads((source / "protocol.json").read_text())
    trainval = np.load(protocol["trainval_npz"], mmap_mode="r", allow_pickle=False)
    holdout = np.load(protocol["holdout_npz"], mmap_mode="r", allow_pickle=False)
    for split, data, prefix in (("train", trainval, "train"),
                                ("val", trainval, "validation"),
                                ("test", holdout, "test")):
        features = eurosat_features(checkpoint, data[prefix + "_images"], device)
        labels = data[prefix + "_labels"]
        rows = [{"domain": int(d), "pair_id": str(i)}
                for d, i in zip(data[prefix + "_domains"], data[prefix + "_ids"])]
        write_split(out, split, features, labels, rows)
    return 10, "frozen_feature_v1", ["RGB", "Sentinel-1 SAR"]


def _load_vision(checkpoint, classes):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model = VisionEncoder(classes)
    model.load_state_dict(state)
    return model.requires_grad_(False).eval()


def prepare_clevr(source, checkpoint, out, device):
    vocab = json.loads((source / "vocab.json").read_text())
    model = _load_vision(checkpoint, 24)
    packed = {}
    for original in ("train", "val"):
        images = np.load(source / f"{original}_images.npz", allow_pickle=False)["images"]
        rows = json.loads((source / f"{original}_questions.json").read_text())
        unique = vision_features(model, images, device)
        query_features = unique[np.asarray([r["image_local"] for r in rows])]
        packed[original] = (query_features, np.asarray([r["label"] for r in rows]), rows,
                            token_array(rows, vocab))
    write_split(out, "train", *packed["train"][:3], token_ids=packed["train"][3])
    held = packed["val"]
    image_ids = sorted({str(r["image_id"]) for r in held[2]},
                       key=lambda s: hashlib.sha256(("17:" + s).encode()).hexdigest())
    val_ids = set(image_ids[::2])
    mask = np.asarray([str(r["image_id"]) in val_ids for r in held[2]])
    for split, keep in (("val", mask), ("test", ~mask)):
        rows = [r for r, selected in zip(held[2], keep) if selected]
        write_split(out, split, held[0][keep], held[1][keep], rows, held[3][keep])
    return 2, "frozen_feature_text_v1", ["RGB image", "attribute-query text"]


def prepare_speech(source, checkpoint, out, device):
    vocab = json.loads((source / "vocab.json").read_text())
    model = _load_vision(checkpoint, 8)
    for split in ("train", "val", "test"):
        images = np.load(source / f"{split}_images.npz", allow_pickle=False)["images"]
        rows = json.loads((source / f"{split}_questions.json").read_text())
        unique = vision_features(model, images, device)
        features = unique[np.asarray([r["image_local"] for r in rows])]
        write_split(out, split, features, [r["label"] for r in rows], rows,
                    token_array(rows, vocab))
    return 2, "frozen_feature_text_v1", ["log-mel audio", "keyword text"]


@torch.no_grad()
def prepare_physical(source, checkpoint, out, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model = PhysicalEncoder(); model.load_state_dict(state); model.eval().to(device)
    query_tokens = np.zeros((2, 32), dtype=np.uint8)
    query_tokens[0, :6] = [2, 3, 4, 5, 6, 7]
    query_tokens[1, :6] = [2, 3, 4, 5, 8, 7]
    for split in ("train", "val", "test"):
        z = np.load(source / f"{split}.npz", mmap_mode="r", allow_pickle=False)
        base_features = []
        for x in batches(z["fields"]):
            value = torch.as_tensor(np.array(x, copy=True), device=device)
            base_features.append(model(value)[0].cpu().numpy().astype(np.float16))
        base_features = np.concatenate(base_features)
        base_labels = np.asarray(z["labels"], dtype=np.int64)
        base_rows = json.loads((source / f"{split}_records.json").read_text())
        features = np.repeat(base_features, 2, axis=0)
        tokens_value = np.tile(query_tokens, (len(base_features), 1))
        labels = (np.tile(np.arange(2), len(base_labels)) == np.repeat(base_labels, 2)).astype(np.int64)
        rows = [{**row, "query": query} for row in base_rows
                for query in ("impossible", "possible")]
        write_split(out, split, features, labels, rows, tokens_value)
    return 2, "frozen_feature_text_v1", ["ordered video frames", "possible/impossible text"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["eurosat", "clevr", "speech", "physical"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    result = globals()["prepare_" + a.task](a.source, a.checkpoint, a.out, a.device)
    classes, storage, modalities = result
    source_manifest = a.source / "manifest.json"
    if not source_manifest.exists() and (a.source.parent / "manifest.json").exists():
        source_manifest = a.source.parent / "manifest.json"
    protocol = {"task": a.task, "classes": classes, "storage": storage,
                "modalities": modalities, "frozen_frontend": str(a.checkpoint.resolve()),
                "frozen_frontend_sha256": sha256(a.checkpoint),
                "source_manifest_sha256": sha256(source_manifest),
                "all_original_samples": False,
                "status": "existing audited preliminary package with shared frozen frontend"}
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {x.name: sha256(x) for x in a.out.iterdir() if x.is_file()}
    (a.out / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
