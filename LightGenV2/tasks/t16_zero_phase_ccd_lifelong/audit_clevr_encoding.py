"""Audit the fixed CLEVR image/text field without training or reading test labels."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from LightGenV2.tasks.t14_shared_readout_lifelong.data import ClevrRawPairs


def audit(source: Path, split: str, count: int):
    data = ClevrRawPairs(source, split)
    vocab = {int(value): key for key, value in json.loads(
        (source / "vocab.json").read_text()).items()}
    count = min(count, len(data) // 2)
    pairs = np.arange(count, dtype=np.int64)
    chosen = np.column_stack((2 * pairs, 2 * pairs + 1)).reshape(-1)
    fields = data.get_batch(chosen, "cpu").numpy().reshape(count, 2, 224, 224)
    images_equal = np.all(fields[:, 0, :112, :] == fields[:, 1, :112, :], axis=(1, 2))
    text_distance = np.linalg.norm(
        (fields[:, 0, 112:, :] - fields[:, 1, 112:, :]).reshape(count, -1), axis=1)
    power = (fields ** 2).sum(axis=(2, 3))
    image_power = (fields[:, :, :112, :] ** 2).sum(axis=(2, 3))
    signatures = defaultdict(lambda: [0, 0])
    for idx in chosen:
        token_ids = data.token_ids[idx]
        signature = tuple(int(item) for item in token_ids if item)
        signatures[signature][int(data.labels[idx])] += 1
    examples = []
    for pair in pairs[:5]:
        examples.append({
            "image_index": int(data.image_index[2 * pair]),
            "positive": " ".join(vocab[int(t)] for t in data.token_ids[2 * pair] if t),
            "negative": " ".join(vocab[int(t)] for t in data.token_ids[2 * pair + 1] if t),
            "text_field_l2_difference": float(text_distance[pair]),
        })
    return {
        "split": split, "total_pairs": len(data) // 2, "audited_pairs": count,
        "labels_per_pair": np.unique(data.labels[chosen].reshape(-1, 2), axis=0).tolist(),
        "paired_image_field_identical_fraction": float(images_equal.mean()),
        "positive_negative_text_field_l2_min_mean_max": [float(text_distance.min()),
            float(text_distance.mean()), float(text_distance.max())],
        "total_input_power_min_mean_max": [float(power.min()), float(power.mean()),
            float(power.max())],
        "image_input_power_min_mean_max": [float(image_power.min()),
            float(image_power.mean()), float(image_power.max())],
        "distinct_text_queries": len(signatures),
        "queries_with_both_labels": sum(all(values) for values in signatures.values()),
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    print(json.dumps(audit(args.source, args.split, args.count), indent=2))


if __name__ == "__main__":
    main()
