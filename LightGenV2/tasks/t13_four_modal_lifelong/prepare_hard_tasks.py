"""Build non-saturating Speech and Physical task wrappers without label-trained frontends."""
import argparse
import json
import os
import shutil
from pathlib import Path

from .data import sha256


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_manifest(out, sources):
    files = {str(path.relative_to(out)): sha256(path)
             for path in out.rglob("*") if path.is_file()}
    (out / "manifest.json").write_text(json.dumps(
        {"files": files, "sources": sources}, indent=2) + "\n")


def speech(source, out):
    protocol_path = source / "protocol.json"
    wrapper = json.loads(protocol_path.read_text())
    assets = Path(wrapper.get("source_root", source))
    target_assets = out / "assets"
    target_assets.mkdir(parents=True)
    vocab = json.loads((assets / "vocab.json").read_text())
    for split in ("train", "val", "test"):
        link_or_copy(assets / f"{split}_images.npz", target_assets / f"{split}_images.npz")
        link_or_copy(assets / f"{split}_questions.json",
                     target_assets / f"{split}_questions.json")
    (target_assets / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    protocol = {
        "task": "speech", "classes": 8,
        "storage": "speech_commands_text_rank8_v2",
        "source_root": str(target_assets.resolve()),
        "modalities": ["raw log-mel audio", "keyword candidate text"],
        "objective": "select the correct text keyword from all eight candidates",
        "primary_metric": "balanced_accuracy", "candidates_per_group": 8,
        "all_original_samples": bool(wrapper.get("all_original_samples", False)),
        "license": "CC BY 4.0", "label_supervised_frontend": False,
        "source_protocol_sha256": sha256(protocol_path),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_manifest(out, {"protocol.json": sha256(protocol_path)})


def physical(source, out):
    concepts = ("continuity", "directional_inertia", "object_persistence",
                "solidity", "unchangeableness")
    roots = {concept: source / concept for concept in concepts}
    protocols = {concept: json.loads((root / "protocol.json").read_text())
                 for concept, root in roots.items()}
    for concept, protocol_value in protocols.items():
        if not protocol_value.get("all_original_samples", False):
            raise ValueError(f"{concept} is not a complete official probe")
    protocol = {
        "task": "physical", "classes": 10,
        "storage": "physical_video_text_rank10_v3",
        "source_roots": {name: str(root.resolve()) for name, root in roots.items()},
        "modalities": ["deterministic adjacent-frame differences",
                       "five-concept x plausibility text candidate bank"],
        "objective": "select the physical concept and possible/impossible text description",
        "primary_metric": "balanced_accuracy",
        "all_original_samples": True,
        "source_quadruplets": sum(int(value["source_quadruplets"])
                                   for value in protocols.values()),
        "concepts": list(concepts), "candidates": 10,
        "license": "CC BY 4.0", "label_supervised_frontend": False,
        "source_protocol_sha256": {
            name: sha256(root / "protocol.json") for name, root in roots.items()},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_manifest(out, {f"{name}/manifest.json": sha256(root / "manifest.json")
                         for name, root in roots.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("speech", "physical"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    globals()[args.task](args.source, args.out)


if __name__ == "__main__":
    main()
