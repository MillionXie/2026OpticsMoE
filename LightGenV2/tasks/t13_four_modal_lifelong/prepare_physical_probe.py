"""Vectorized full Physical Concepts probe preparation for the 10-way task."""
import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from LightGenV2.tasks.t12_cross_modal_lifelong.prepare_physical_concepts import (
    FRAME_INDEX, LICENSE_PAGE, OFFICIAL_ROOT, rank, records, sha,
)


def encode_batch(videos):
    x = torch.from_numpy(np.stack(videos)[:, FRAME_INDEX].astype(np.float32) / 255.0)
    x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    batch = len(x)
    x = F.interpolate(x.reshape(batch * 8, 1, 64, 64), (112, 56),
                      mode="bilinear", align_corners=False)[:, 0]
    x = x.reshape(batch, 2, 4, 112, 56).permute(0, 1, 3, 2, 4)
    field = x.reshape(batch, 224, 224)
    field = field / field.square().sum((-2, -1), keepdim=True).clamp_min(1e-20).sqrt()
    return field.numpy().astype(np.float16)


def flush(pending, result):
    if not pending:
        return
    fields = encode_batch([item[1] for item in pending])
    for field, (split, _, label, row) in zip(fields, pending):
        result[split][0].append(field)
        result[split][1].append(label)
        result[split][2].append(row)
    pending.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=False)
    paths = []
    for index in range(20):
        name = f"data.tfrecord-{index:05d}-of-00020"
        path = args.cache / args.concept / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(f"{OFFICIAL_ROOT}/{args.concept}/{name}", path)
        paths.append(path)
    candidates = []
    for shard, path in enumerate(paths):
        for row, record in enumerate(records(path)):
            candidates.append((rank(f"{args.concept}:{shard}:{row}"), shard, row, record))
    selected = sorted(candidates, key=lambda value: value[0])
    if len(selected) != 5000:
        raise ValueError(f"expected 5000 quadruplets, found {len(selected)}")
    result = {split: ([], [], []) for split in ("train", "val", "test")}
    counts = Counter(); pending = []
    for _, shard, row, record in selected:
        identity = f"{args.concept}:{shard}:{row}"
        bucket = int(rank(identity, "split")[:8], 16) / 0xffffffff
        split = "train" if bucket < .70 else ("val" if bucket < .85 else "test")
        counts[split] += 1
        for kind, label in (("possible", 1), ("impossible", 0)):
            raw = record.features.feature[f"{kind}_image"].bytes_list.value[0]
            videos = np.frombuffer(raw, dtype=np.uint8).reshape(2, 15, 64, 64, 3)
            for pair in range(2):
                pending.append((split, videos[pair].copy(), label, {
                    "concept": args.concept, "quadruplet": f"{shard}:{row}",
                    "kind": kind, "pair": pair, "label": label,
                }))
                if len(pending) >= args.batch:
                    flush(pending, result)
    flush(pending, result)
    for split, (fields, labels, rows) in result.items():
        np.savez_compressed(args.out / f"{split}.npz", fields=np.stack(fields),
                            labels=np.asarray(labels, dtype=np.int64))
        (args.out / f"{split}_records.json").write_text(json.dumps(rows, indent=2) + "\n")
    ids = {split: {row["quadruplet"] for row in result[split][2]} for split in result}
    assert all(not ids[a] & ids[b] for i, a in enumerate(ids)
               for b in list(ids)[i + 1:])
    protocol = {
        "task": "video", "classes": 2, "dataset": "Physical Concepts Dataset",
        "concept": args.concept, "scope": f"complete {args.concept} probe",
        "all_original_samples": True, "source_quadruplets": 5000,
        "input": "8 ordered luminance frames tiled 2x4 into 224x224; unit total power",
        "label": {"impossible": 0, "possible": 1},
        "split": "quadruplet-disjoint hash 70/15/15", "quadruplets": dict(counts),
        "license": "CC BY 4.0", "license_page": LICENSE_PAGE,
        "source": f"{OFFICIAL_ROOT}/{args.concept}/data.tfrecord-{{00000..00019}}-of-00020",
        "raw_shards": {path.name: sha(path) for path in paths},
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {path.name: sha(path) for path in args.out.iterdir() if path.is_file()}
    (args.out / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
