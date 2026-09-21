"""Create small immutable reference packages around existing audited assets."""
import argparse
import json
import io
import os
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np

from LightGenV2.tasks.t09_multimodal_matching.audio_prepare import WORDS, logmel

from .data import sha256


TASKS = {
    "eurosat": (10, "eurosat_rgb_sar_v1"),
    "clevr": (2, "npz_fields_v1"),
    "speech": (2, "speech_commands_text_v1"),
    "physical": (2, "physical_video_text_v1"),
}


def write_package(out, protocol):
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    manifest = {"files": {"protocol.json": sha256(out / "protocol.json")},
                "sources": protocol.get("source_hashes", {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--holdout", type=Path)
    p.add_argument("--archive", type=Path,
                   help="mini_speech_commands.zip, required when the held-out audio is not materialized")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    classes, storage = TASKS[a.task]
    protocol = {"task": a.task, "classes": classes, "storage": storage,
                "all_original_samples": False,
                "status": "existing audited preliminary package"}
    if a.task == "eurosat":
        if a.holdout is None:
            raise ValueError("EuroSAT requires --holdout")
        trainval, holdout = a.source / "data.npz", a.holdout / "data.npz"
        protocol.update(trainval_npz=str(trainval.resolve()), holdout_npz=str(holdout.resolve()),
                        modalities=["RGB", "Sentinel-1 SAR"], objective="10-class land-cover classification",
                        source_hashes={str(trainval): sha256(trainval), str(holdout): sha256(holdout)},
                        license="MIT (dataset release; explicit exception to the earlier CC-BY-only preference)")
    elif a.task == "speech":
        # Keep the original package immutable. Build a small complete view with
        # hard links for train/val and decode the reserved speaker-disjoint test.
        a.out.mkdir(parents=True, exist_ok=False)
        assets = a.out / "assets"; assets.mkdir()
        for name in ("train_images.npz", "val_images.npz", "train_questions.json",
                     "val_questions.json", "vocab.json"):
            source, target = a.source / name, assets / name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        source_manifest = json.loads((a.source / "manifest.json").read_text())
        records = json.loads((a.source / "test_reserved_ids.json").read_text())
        if a.archive is None:
            raise ValueError("Speech Commands requires --archive to decode the reserved test speakers")
        images, rows = [], []
        with zipfile.ZipFile(a.archive) as archive:
            for i, record in enumerate(records):
                with wave.open(io.BytesIO(archive.read(record["source_member"])), "rb") as stream:
                    samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
                image = logmel(samples)
                images.append(np.repeat(image[:, :, None], 3, axis=2))
                target = WORDS.index(record["word"])
                offset = 1 + int(record["sha256"][:8], 16) % 7
                for label, query in ((1, target), (0, (target + offset) % 8)):
                    rows.append({"image_local": i, "image_id": record["source_member"],
                                 "speaker": record["speaker"],
                                 "question": "does the audio say " + WORDS[query] + " ?",
                                 "label": label, "audio_class": target, "query_class": query})
        np.savez_compressed(assets / "test_images.npz", images=np.stack(images))
        (assets / "test_questions.json").write_text(json.dumps(rows, indent=2) + "\n")
        retained_counts = {split: sum(source_manifest["counts"][split].values())
                           for split in ("train", "val", "test")}
        protocol.update(all_original_samples=True,
                        status="complete deduplicated official mini_speech_commands release",
                        source_release="mini_speech_commands",
                        retained_audio_clips=retained_counts,
                        duplicate_waveforms_removed=source_manifest["duplicate_waveforms_removed"],
                        query_counts={split: 2 * count for split, count in retained_counts.items()},
                        source_root=str(assets.resolve()), modalities=["log-mel audio", "keyword text"],
                        objective="balanced audio/text keyword matching", license="CC BY 4.0",
                        source_hashes={"manifest.json": sha256(a.source / "manifest.json"),
                                       "archive": sha256(a.archive)})
        (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        files = {str(x.relative_to(a.out)): sha256(x) for x in a.out.rglob("*") if x.is_file()}
        (a.out / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
        return
    elif a.task == "physical":
        protocol.update(source_root=str(a.source.resolve()), modalities=["ordered video frames", "possible/impossible text"],
                        objective="balanced video/text physical-plausibility matching", license="CC BY 4.0",
                        source_hashes={"manifest.json": sha256(a.source / "manifest.json")})
    else:
        # CLEVR fields are copied as immutable links/files into the new package.
        import shutil
        a.out.mkdir(parents=True, exist_ok=False)
        for split in ("train", "val", "test"):
            for suffix in (".npz", "_records.json"):
                source = a.source / f"{split}{suffix}"
                if source.exists():
                    shutil.copy2(source, a.out / source.name)
        protocol.update(modalities=["RGB image", "attribute-query text"],
                        objective="balanced image/text attribute matching", license="CC BY 4.0",
                        source_hashes={"manifest.json": sha256(a.source / "manifest.json")})
        (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        files = {x.name: sha256(x) for x in a.out.iterdir() if x.is_file()}
        (a.out / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
        return
    write_package(a.out, protocol)


if __name__ == "__main__":
    main()
