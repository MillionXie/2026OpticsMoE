"""Build non-saturating Speech and Physical task wrappers without label-trained frontends."""
import argparse
import json
import os
import shutil
from pathlib import Path

from LightGenV2.tasks.t09_multimodal_matching.audio_prepare import WORDS

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
    source_protocol = source / "protocol.json"
    wrapper = json.loads(source_protocol.read_text())
    raw = Path(wrapper.get("source_root", source))
    protocol = {
        "task": "physical", "classes": 2,
        "storage": "physical_video_text_delta_v2", "source_root": str(raw.resolve()),
        "modalities": ["deterministic adjacent-frame differences", "possible/impossible text"],
        "objective": "video/text physical-plausibility matching",
        "primary_metric": "balanced_accuracy",
        "all_original_samples": bool(wrapper.get("all_original_samples", False)),
        "license": "CC BY 4.0", "label_supervised_frontend": False,
        "source_protocol_sha256": sha256(source_protocol),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_manifest(out, {"protocol.json": sha256(source_protocol),
                         "raw_manifest.json": sha256(raw / "manifest.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("speech", "physical"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    globals()[args.task](args.source, args.out)


if __name__ == "__main__":
    main()
