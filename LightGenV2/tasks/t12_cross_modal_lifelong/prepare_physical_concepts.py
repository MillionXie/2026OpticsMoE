"""Prepare a deterministic CC-BY-4.0 video plausibility subset."""
import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tfrecord import tfrecord_loader


OFFICIAL_ROOT = "https://storage.googleapis.com/physical_concepts/probes"
LICENSE_PAGE = "https://github.com/google-deepmind/physical_concepts#license-and-disclaimer"
FRAME_INDEX = np.linspace(0, 14, 8).round().astype(int)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(value):
    return hashlib.sha256(("17:" + value).encode()).hexdigest()


def encode_video(video):
    """Eight ordered luminance frames -> 2x4 optical tile field."""
    x = torch.from_numpy(video[FRAME_INDEX].astype(np.float32) / 255.0)
    x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    x = F.interpolate(x[:, None], (112, 56), mode="bilinear", align_corners=False)[:, 0]
    field = torch.cat([torch.cat(list(x[r*4:(r+1)*4]), -1) for r in range(2)], -2)
    return (field / field.square().sum().clamp_min(1e-20).sqrt()).numpy().astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concept", default="continuity")
    p.add_argument("--max-quadruplets", type=int, default=1024)
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=False)
    paths = []
    for i in range(20):
        name = f"data.tfrecord-{i:05d}-of-00020"
        path = a.cache / a.concept / name; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(f"{OFFICIAL_ROOT}/{a.concept}/{name}", path)
        paths.append(path)
    description = {"possible_image": "byte", "impossible_image": "byte"}
    candidates = []
    for shard, path in enumerate(paths):
        for row, record in enumerate(tfrecord_loader(str(path), None, description, compression_type="gzip")):
            candidates.append((rank(f"{a.concept}:{shard}:{row}"), shard, row, record))
    selected = sorted(candidates, key=lambda x: x[0])[:a.max_quadruplets]
    result = {s:([], [], []) for s in ("train", "val", "test")}
    split_counts = Counter()
    for digest, shard, row, record in selected:
        bucket = int(digest[:8], 16) / 0xffffffff
        split = "train" if bucket < .70 else ("val" if bucket < .85 else "test")
        split_counts[split] += 1
        for kind, label in (("possible", 1), ("impossible", 0)):
            raw = record[f"{kind}_image"]
            if isinstance(raw, np.ndarray): raw = raw.reshape(-1)[0]
            videos = np.frombuffer(raw, dtype=np.uint8).reshape(2, 15, 64, 64, 3)
            for pair in range(2):
                field = encode_video(videos[pair])
                result[split][0].append(field); result[split][1].append(label)
                result[split][2].append({"concept":a.concept,"quadruplet":f"{shard}:{row}","kind":kind,"pair":pair,"label":label})
    for split, (fields, labels, rows) in result.items():
        assert fields
        np.savez_compressed(a.out/f"{split}.npz", fields=np.stack(fields), labels=np.asarray(labels,dtype=np.int64))
        (a.out/f"{split}_records.json").write_text(json.dumps(rows,indent=2)+"\n")
    ids={s:{r["quadruplet"] for r in result[s][2]} for s in result}
    assert all(not ids[a]&ids[b] for i,a in enumerate(ids) for b in list(ids)[i+1:])
    protocol={"task":"video","classes":2,"dataset":"Physical Concepts Dataset","concept":a.concept,
              "scope":f"deterministic seed-17 subset of {len(selected)} probe quadruplets",
              "input":"8 ordered luminance frames at indices "+str(FRAME_INDEX.tolist())+" tiled 2x4 into 224x224; unit total power",
              "label":{"impossible":0,"possible":1},"split":"quadruplet-disjoint hash 70/15/15",
              "quadruplets":dict(split_counts),"license":"CC BY 4.0","license_page":LICENSE_PAGE,
              "source":f"{OFFICIAL_ROOT}/{a.concept}/data.tfrecord-{{00000..00019}}-of-00020",
              "raw_shards":{p.name:sha(p) for p in paths}}
    (a.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    files={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}
    (a.out/"manifest.json").write_text(json.dumps(files,indent=2)+"\n")


if __name__ == "__main__":
    main()
