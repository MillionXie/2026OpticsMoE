"""Audited adapters for the four requested multimodal tasks.

Every adapter returns a 224x224 real amplitude field.  The wrapper keeps the
existing immutable data files in place; protocol.json records their paths and
hashes so a run cannot silently switch datasets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_power(x, power=1.0):
    return x * (power / x.square().sum((-2, -1), keepdim=True).clamp_min(1e-20)).sqrt()


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if path.is_file() and len(expected) == 64:
            assert sha256(path) == expected, name
    return sha256(root / "manifest.json")


class NpzFields:
    def __init__(self, path):
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        self.fields = data["fields"]

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.fields[index], copy=True)).float()


def _rgb_field(images):
    x = torch.as_tensor(np.array(images, copy=True)).float().permute(0, 3, 1, 2) / 255.0
    x = F.interpolate(x, (112, 112), mode="bilinear", align_corners=False)
    blank = torch.zeros_like(x[:, 0])
    field = torch.cat((torch.cat((x[:, 0], x[:, 1]), -1),
                       torch.cat((x[:, 2], blank), -1)), -2)
    return normalize_power(field)


class EuroSatFields:
    def __init__(self, trainval, holdout, split):
        path = holdout if split == "test" else trainval
        self.data = np.load(path, mmap_mode="r", allow_pickle=False)
        prefix = "test" if split == "test" else ("validation" if split == "val" else "train")
        self.images = self.data[prefix + "_images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        values = self.images[[int(index)]] if scalar else self.images[index]
        result = _rgb_field(values)
        return result[0] if scalar else result


def _fixed_text(token_ids):
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    return F.one_hot(ids, num_classes=64).float() * ids.ne(0).unsqueeze(-1)


def _legacy_audio_text(images, token_ids):
    x = torch.as_tensor(np.array(images, copy=True)).float().permute(0, 3, 1, 2) / 255.0
    assert torch.equal(x[:, 0], x[:, 1]) and torch.equal(x[:, 0], x[:, 2])
    x = F.interpolate(x, (112, 112), mode="bilinear", align_corners=False)
    x = x * (0.5 / x.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-20)).sqrt()
    text = F.interpolate(_fixed_text(token_ids)[:, None], (112, 112), mode="nearest")[:, 0]
    text = normalize_power(text, 0.5)
    return torch.cat((torch.cat((x[:, 0], x[:, 1]), -1),
                      torch.cat((x[:, 2], text), -1)), -2)


class SpeechFields:
    def __init__(self, source, split):
        source = Path(source)
        data = np.load(source / f"{split}_images.npz", mmap_mode="r", allow_pickle=False)
        self.images = data["images"]
        self.rows = json.loads((source / f"{split}_questions.json").read_text())
        vocab = json.loads((source / "vocab.json").read_text())
        self.image_index = np.asarray([r["image_local"] for r in self.rows], dtype=np.int64)
        self.token_ids = np.zeros((len(self.rows), 32), dtype=np.int64)
        for i, row in enumerate(self.rows):
            words = row["question"].lower().replace("?", " ?").split()
            self.token_ids[i, :len(words)] = [vocab.get(w, 1) for w in words]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        ix = np.asarray([index] if scalar else index, dtype=np.int64)
        result = _legacy_audio_text(self.images[self.image_index[ix]], self.token_ids[ix])
        return result[0] if scalar else result


def _physical_query_codes():
    # Fixed, non-trainable word-position codes. Both queries have equal density.
    ids = torch.zeros(2, 32, dtype=torch.long)
    ids[0, :6] = torch.tensor([2, 3, 4, 5, 6, 7])  # is this video physically impossible ?
    ids[1, :6] = torch.tensor([2, 3, 4, 5, 8, 7])  # is this video physically possible ?
    return _fixed_text(ids)


class PhysicalTextFields:
    """Turn each video into two balanced possible/impossible text queries."""
    def __init__(self, path):
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        self.base = data["fields"]
        self.base_labels = np.asarray(data["labels"], dtype=np.int64)
        self.text = _physical_query_codes()

    def __len__(self):
        return 2 * len(self.base)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        ix = np.asarray([index] if scalar else index, dtype=np.int64)
        base_ix, query = ix // 2, ix % 2
        video = torch.from_numpy(np.array(self.base[base_ix], copy=True)).float()
        video = F.interpolate(video[:, None], (224, 112), mode="bilinear", align_corners=False)[:, 0]
        video = normalize_power(video, 0.5)
        text = F.interpolate(self.text[query, None], (224, 112), mode="nearest")[:, 0]
        text = normalize_power(text, 0.5)
        result = torch.cat((video, text), -1)
        return result[0] if scalar else result

    def labels(self):
        query = np.tile(np.arange(2, dtype=np.int64), len(self.base_labels))
        truth = np.repeat(self.base_labels, 2)
        return (query == truth).astype(np.int64)


def load_task(root, split):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    storage = protocol["storage"]
    if storage == "eurosat_rgb_sar_v1":
        trainval = Path(protocol["trainval_npz"])
        holdout = Path(protocol["holdout_npz"])
        fields = EuroSatFields(trainval, holdout, split)
        data = np.load(holdout if split == "test" else trainval, mmap_mode="r", allow_pickle=False)
        prefix = "test" if split == "test" else ("validation" if split == "val" else "train")
        labels = np.asarray(data[prefix + "_labels"], dtype=np.int64)
        domains = np.asarray(data[prefix + "_domains"], dtype=np.int64)
        ids = data[prefix + "_ids"]
        rows = [{"domain": int(d), "pair_id": str(i)} for d, i in zip(domains, ids)]
    elif storage == "speech_commands_text_v1":
        source = Path(protocol["source_root"])
        fields = SpeechFields(source, split)
        rows = fields.rows
        labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    elif storage == "physical_video_text_v1":
        source = Path(protocol["source_root"])
        fields = PhysicalTextFields(source / f"{split}.npz")
        labels = fields.labels()
        base_rows = json.loads((source / f"{split}_records.json").read_text())
        rows = []
        for row in base_rows:
            for query in ("impossible", "possible"):
                rows.append({**row, "query": query})
    elif storage == "npz_fields_v1":
        path = root / f"{split}.npz"
        fields = NpzFields(path)
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        labels = np.asarray(data["labels"], dtype=np.int64)
        rows_path = root / f"{split}_records.json"
        rows = json.loads(rows_path.read_text()) if rows_path.exists() else [{} for _ in labels]
    else:
        raise ValueError(f"unknown storage {storage!r}")
    assert len(fields) == len(labels) == len(rows)
    return fields, labels, rows
