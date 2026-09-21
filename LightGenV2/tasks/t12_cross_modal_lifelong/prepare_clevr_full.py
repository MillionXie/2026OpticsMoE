"""Prepare all labeled CLEVR images for the derived color-shape query task."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from LightGenV2.tasks.t09_multimodal_matching.prepare import COLORS, SHAPES, TEMPLATES, tokens


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ranked(ids, seed=17):
    return sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())


def queries(scene, seed):
    combinations = {(color, shape) for color in COLORS for shape in SHAPES}
    present = {(obj["color"], obj["shape"]) for obj in scene["objects"]}
    rng = random.Random(seed * 100000 + int(scene["image_index"]))
    positive = sorted(present)
    selected_positive = rng.sample(positive, min(3, len(positive)))
    while len(selected_positive) < 3:
        selected_positive.append(rng.choice(positive))
    selected_negative = rng.sample(sorted(combinations - present), 3)
    for pair, (yes, no) in enumerate(zip(selected_positive, selected_negative)):
        template = TEMPLATES[(int(scene["image_index"]) + pair) % len(TEMPLATES)]
        for label, (color, shape) in ((1, yes), (0, no)):
            yield template.format(color=color, shape=shape), label


def prepare_split(root, out, split, scenes, vocab, seed):
    image_dir = root / "images" / ("train" if split == "train" else "val")
    images = np.lib.format.open_memmap(out / f"{split}_images.npy", mode="w+",
                                       dtype=np.uint8, shape=(len(scenes), 64, 64, 3))
    count = len(scenes) * 6
    image_index = np.lib.format.open_memmap(out / f"{split}_image_index.npy", mode="w+",
                                            dtype=np.int32, shape=(count,))
    token_ids = np.lib.format.open_memmap(out / f"{split}_token_ids.npy", mode="w+",
                                          dtype=np.uint8, shape=(count, 32))
    labels = np.lib.format.open_memmap(out / f"{split}_labels.npy", mode="w+",
                                       dtype=np.int64, shape=(count,))
    cursor = 0
    for local, scene in enumerate(scenes):
        path = image_dir / scene["image_filename"]
        images[local] = np.asarray(Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS))
        for question, label in queries(scene, seed):
            words = tokens(question)
            image_index[cursor] = local
            token_ids[cursor, :len(words)] = [vocab.get(word, 1) for word in words]
            labels[cursor] = label
            cursor += 1
        if (local + 1) % 5000 == 0:
            print(json.dumps({"split": split, "images": local + 1, "queries": cursor}), flush=True)
    images.flush(); image_index.flush(); token_ids.flush(); labels.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted CLEVR_v1.0 directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    root, out = args.source, args.out
    out.mkdir(parents=True, exist_ok=False)
    train = json.loads((root / "scenes/CLEVR_train_scenes.json").read_text())["scenes"]
    held = json.loads((root / "scenes/CLEVR_val_scenes.json").read_text())["scenes"]
    if len(train) != 70000 or len(held) != 15000:
        raise ValueError(f"expected 70,000/15,000 scenes, found {len(train)}/{len(held)}")
    held_order = ranked([int(row["image_index"]) for row in held], args.seed)
    val_ids = set(held_order[:7500])
    val = [row for row in held if int(row["image_index"]) in val_ids]
    test = [row for row in held if int(row["image_index"]) not in val_ids]
    vocab = {"<pad>": 0, "<unk>": 1}
    for template in TEMPLATES:
        for color in COLORS:
            for shape in SHAPES:
                for word in tokens(template.format(color=color, shape=shape)):
                    vocab.setdefault(word, len(vocab))
    if len(vocab) > 64:
        raise ValueError("fixed text encoder supports at most 64 tokens")
    for split, scenes in (("train", train), ("val", val), ("test", test)):
        prepare_split(root, out, split, scenes, vocab, args.seed)
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    protocol = {
        "task": "clevr", "classes": 2, "dataset": "CLEVR v1.0",
        "storage": "clevr_lazy_v1", "all_original_samples": True,
        "source_train_images": 70000, "source_val_images": 15000,
        "split_images": {"train": 70000, "val": 7500, "test": 7500},
        "split_queries": {"train": 420000, "val": 45000, "test": 45000},
        "scope": "all train and validation images with public scene graphs; six deterministic balanced derived queries per image",
        "test_policy": "official test excluded because answers and scene graphs are not public; official validation split image-disjoint 50/50",
        "input": "RGB [R,G;B,zero] left half and fixed token-position one-hot right half; power 0.5 each",
        "license": "CC BY 4.0", "source": "https://cs.stanford.edu/people/jcjohns/clevr/",
        "seed": args.seed,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
