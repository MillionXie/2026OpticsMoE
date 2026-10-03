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


def _candidate_text_bank(token_rows):
    """Stack fixed text candidates vertically into a 224x112 optical field."""
    encoded = _fixed_text(token_rows)
    count = len(encoded)
    edges = np.linspace(0, 224, count + 1).round().astype(int)
    bands = [F.interpolate(encoded[i][None, None],
                           (int(edges[i + 1] - edges[i]), 112), mode="nearest")[0, 0]
             for i in range(count)]
    return normalize_power(torch.cat(bands, 0), 0.5)


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


class SpeechRank8Fields:
    """One utterance plus a fixed bank of all eight candidate word codes."""
    def __init__(self, source, split):
        source = Path(source)
        data = np.load(source / f"{split}_images.npz", mmap_mode="r", allow_pickle=False)
        self.images = data["images"]
        paired = json.loads((source / f"{split}_questions.json").read_text())
        by_image = {}
        for row in paired:
            by_image.setdefault(int(row["image_local"]), row)
        self.rows = [{"image_local": image_local, "image_id": row["image_id"],
                      "speaker": row["speaker"], "audio_class": int(row["audio_class"]),
                      "candidate_words": ["down", "go", "left", "no",
                                          "right", "stop", "up", "yes"]}
                     for image_local, row in sorted(by_image.items())]
        # Eight equal-height bands form a fixed text candidate bank. Each band
        # contains a deterministic binary code for one word; no learned text
        # encoder or label-trained audio frontend is present.
        tokens = torch.zeros(8, 4, dtype=torch.long)
        tokens[:, 0] = torch.arange(2, 10)
        self.candidate_bank = _candidate_text_bank(tokens)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        ix = np.asarray([index] if scalar else index, dtype=np.int64)
        image_ix = np.asarray([self.rows[int(i)]["image_local"] for i in ix])
        audio = torch.as_tensor(np.array(self.images[image_ix], copy=True)).float()[..., 0] / 255.0
        audio = F.interpolate(audio[:, None], (224, 112), mode="bilinear", align_corners=False)[:, 0]
        audio = normalize_power(audio, 0.5)
        bank = self.candidate_bank.expand(len(ix), -1, -1)
        result = torch.cat((audio, bank), -1)
        return result[0] if scalar else result


def _physical_query_codes():
    # Fixed, non-trainable word-position codes. Both queries have equal density.
    ids = torch.zeros(2, 32, dtype=torch.long)
    ids[0, :6] = torch.tensor([2, 3, 4, 5, 6, 7])  # is this video physically impossible ?
    ids[1, :6] = torch.tensor([2, 3, 4, 5, 8, 7])  # is this video physically possible ?
    return _fixed_text(ids)


class PhysicalTextFields:
    """Turn each video into two balanced possible/impossible text queries."""
    def __init__(self, path, temporal_delta=False):
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        self.base = data["fields"]
        self.base_labels = np.asarray(data["labels"], dtype=np.int64)
        self.text = _physical_query_codes()
        self.temporal_delta = bool(temporal_delta)

    def __len__(self):
        return 2 * len(self.base)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        ix = np.asarray([index] if scalar else index, dtype=np.int64)
        base_ix, query = ix // 2, ix % 2
        video = torch.from_numpy(np.array(self.base[base_ix], copy=True)).float()
        if self.temporal_delta:
            # The source stores eight ordered 112x56 frames in a 2x4 mosaic.
            # Adjacent differences expose motion without a learned electronic
            # video encoder and, crucially, without training on the target.
            frames = video.reshape(-1, 2, 112, 4, 56).permute(
                0, 1, 3, 2, 4).reshape(-1, 8, 112, 56)
            delta = frames[:, 1:] - frames[:, :-1]
            delta = torch.cat((delta, torch.zeros_like(delta[:, :1])), 1)
            video = delta.reshape(-1, 2, 4, 112, 56).permute(
                0, 1, 3, 2, 4).reshape(-1, 224, 224)
        video = F.interpolate(video[:, None], (224, 112), mode="bilinear", align_corners=False)[:, 0]
        video = normalize_power(video, 0.5)
        text = F.interpolate(self.text[query, None], (224, 112), mode="nearest")[:, 0]
        text = normalize_power(text, 0.5)
        result = normalize_power(torch.cat((video, text), -1))
        return result[0] if scalar else result

    def labels(self):
        query = np.tile(np.arange(2, dtype=np.int64), len(self.base_labels))
        truth = np.repeat(self.base_labels, 2)
        return (query == truth).astype(np.int64)


class PhysicalRank10Fields:
    """Five physical concepts x possible/impossible, with no learned frontend."""
    CONCEPTS = ("continuity", "directional_inertia", "object_persistence",
                "solidity", "unchangeableness")

    def __init__(self, roots, split):
        self.fields = []
        self.base_labels = []
        self.rows = []
        self.offsets = [0]
        for concept_index, concept in enumerate(self.CONCEPTS):
            root = Path(roots[concept])
            data = np.load(root / f"{split}.npz", mmap_mode="r", allow_pickle=False)
            fields = data["fields"]
            labels = np.asarray(data["labels"], dtype=np.int64)
            records = json.loads((root / f"{split}_records.json").read_text())
            assert len(fields) == len(labels) == len(records)
            self.fields.append(fields); self.base_labels.append(labels)
            for label, row in zip(labels, records):
                self.rows.append({**row, "concept": concept,
                                  "plausibility": "possible" if int(label) else "impossible",
                                  "candidate_texts": [
                                      f"{name.replace('_', ' ')} {kind}"
                                      for name in self.CONCEPTS
                                      for kind in ("impossible", "possible")]})
            self.offsets.append(self.offsets[-1] + len(labels))
        # concept token, optional second concept token, plausibility token
        tokens = torch.zeros(10, 6, dtype=torch.long)
        for concept in range(5):
            tokens[2 * concept:2 * concept + 2, 0] = 2 + concept
            tokens[2 * concept, 1] = 7
            tokens[2 * concept + 1, 1] = 8
        self.candidate_bank = _candidate_text_bank(tokens)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        indices = np.asarray([index] if scalar else index, dtype=np.int64)
        videos = []
        for index_value in indices:
            concept = int(np.searchsorted(self.offsets, int(index_value), side="right") - 1)
            local = int(index_value) - self.offsets[concept]
            videos.append(np.array(self.fields[concept][local], copy=True))
        video = torch.from_numpy(np.stack(videos)).float()
        frames = video.reshape(-1, 2, 112, 4, 56).permute(
            0, 1, 3, 2, 4).reshape(-1, 8, 112, 56)
        delta = frames[:, 1:] - frames[:, :-1]
        delta = torch.cat((delta, torch.zeros_like(delta[:, :1])), 1)
        video = delta.reshape(-1, 2, 4, 112, 56).permute(
            0, 1, 3, 2, 4).reshape(-1, 224, 224)
        video = F.interpolate(video[:, None], (224, 112), mode="bilinear", align_corners=False)[:, 0]
        video = normalize_power(video, 0.5)
        result = normalize_power(torch.cat(
            (video, self.candidate_bank.expand(len(indices), -1, -1)), -1))
        return result[0] if scalar else result

    def labels(self):
        return np.concatenate([2 * concept + labels
                               for concept, labels in enumerate(self.base_labels)]).astype(np.int64)


def _feature_only_field(features):
    value = torch.as_tensor(np.array(features, copy=True)).float().reshape(-1, 16, 8)
    value = value.repeat_interleave(14, 1).repeat_interleave(28, 2)
    return normalize_power(value)


def _feature_text_field(features, token_ids):
    value = torch.as_tensor(np.array(features, copy=True)).float().reshape(-1, 1, 16, 8)
    value = value.expand(-1, 3, -1, -1)
    value = F.interpolate(value, (112, 112), mode="nearest")
    value = value * (0.5 / value.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-20)).sqrt()
    text = F.interpolate(_fixed_text(token_ids)[:, None], (112, 112), mode="nearest")[:, 0]
    text = normalize_power(text, 0.5)
    return torch.cat((torch.cat((value[:, 0], value[:, 1]), -1),
                      torch.cat((value[:, 2], text), -1)), -2)


class FeatureFields:
    def __init__(self, root, split, with_text):
        self.features = np.load(root / f"{split}_features.npy", mmap_mode="r")
        self.token_ids = (np.load(root / f"{split}_token_ids.npy", mmap_mode="r")
                          if with_text else None)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        ix = np.asarray([index] if scalar else index, dtype=np.int64)
        result = (_feature_text_field(self.features[ix], self.token_ids[ix])
                  if self.token_ids is not None else _feature_only_field(self.features[ix]))
        return result[0] if scalar else result

    def get_batch(self, index, device):
        """Expand compact frozen features directly on the training device.

        Full CLEVR has 420,000 training queries.  Repeating interpolation on
        the CPU for every epoch starves the optical FFTs, while the compact
        arrays are only 128 feature values plus 32 token ids per example.
        """
        if isinstance(index, slice):
            index = np.arange(len(self), dtype=np.int64)[index]
        ix = np.asarray(index, dtype=np.int64)
        features = torch.as_tensor(np.array(self.features[ix], copy=True),
                                   device=device, dtype=torch.float32)
        if self.token_ids is None:
            value = features.reshape(-1, 16, 8)
            value = value.repeat_interleave(14, 1).repeat_interleave(28, 2)
            return normalize_power(value)
        value = features.reshape(-1, 1, 16, 8).expand(-1, 3, -1, -1)
        value = F.interpolate(value, (112, 112), mode="nearest")
        value = value * (0.5 / value.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-20)).sqrt()
        token_ids = torch.as_tensor(np.array(self.token_ids[ix], copy=True),
                                    device=device, dtype=torch.long)
        text = F.interpolate(_fixed_text(token_ids)[:, None], (112, 112), mode="nearest")[:, 0]
        text = normalize_power(text, 0.5)
        return torch.cat((torch.cat((value[:, 0], value[:, 1]), -1),
                          torch.cat((value[:, 2], text), -1)), -2)


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
    elif storage == "speech_commands_text_rank8_v2":
        source = Path(protocol["source_root"])
        fields = SpeechRank8Fields(source, split)
        rows = fields.rows
        labels = np.asarray([r["audio_class"] for r in rows], dtype=np.int64)
    elif storage in {"physical_video_text_v1", "physical_video_text_delta_v2"}:
        source = Path(protocol["source_root"])
        fields = PhysicalTextFields(source / f"{split}.npz",
                                    temporal_delta=storage.endswith("delta_v2"))
        labels = fields.labels()
        base_rows = json.loads((source / f"{split}_records.json").read_text())
        rows = []
        for row in base_rows:
            for query in ("impossible", "possible"):
                rows.append({**row, "query": query})
    elif storage == "physical_video_text_rank10_v3":
        fields = PhysicalRank10Fields(protocol["source_roots"], split)
        labels = fields.labels()
        rows = fields.rows
    elif storage == "npz_fields_v1":
        path = root / f"{split}.npz"
        fields = NpzFields(path)
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        labels = np.asarray(data["labels"], dtype=np.int64)
        rows_path = root / f"{split}_records.json"
        rows = json.loads(rows_path.read_text()) if rows_path.exists() else [{} for _ in labels]
    elif storage in {"frozen_feature_v1", "frozen_feature_text_v1"}:
        fields = FeatureFields(root, split, storage.endswith("_text_v1"))
        labels = np.asarray(np.load(root / f"{split}_labels.npy", mmap_mode="r"), dtype=np.int64)
        rows_path = root / f"{split}_records.json"
        rows = json.loads(rows_path.read_text()) if rows_path.exists() else [{} for _ in labels]
    else:
        raise ValueError(f"unknown storage {storage!r}")
    assert len(fields) == len(labels) == len(rows)
    return fields, labels, rows
