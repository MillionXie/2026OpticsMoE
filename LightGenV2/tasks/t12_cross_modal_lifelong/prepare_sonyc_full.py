"""Prepare every annotated SONYC-UST v2.3 recording without query duplication."""
import argparse
import csv
import hashlib
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from .data import SONYC_EVENTS, normalize_power


EVENT_WORDS = {
    "1_engine_presence": ["engine"],
    "2_machinery-impact_presence": ["machinery", "impact"],
    "3_non-machinery-impact_presence": ["non", "machinery", "impact"],
    "4_powered-saw_presence": ["powered", "saw"],
    "5_alert-signal_presence": ["alert", "signal"],
    "6_music_presence": ["music"],
    "7_human-voice_presence": ["human", "voice"],
    "8_dog_presence": ["dog"],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_label(rows, event, split):
    verified = [int(row[event]) for row in rows
                if row["annotator_id"] == "0" and row[event] in ("0", "1")]
    if verified:
        if len(set(verified)) != 1:
            raise ValueError(f"conflicting verified labels for {rows[0]['audio_filename']} {event}")
        return verified[0], "verified_annotator_0"
    votes = [int(row[event]) for row in rows
             if row["annotator_id"].isdigit() and int(row["annotator_id"]) > 0
             and row[event] in ("0", "1")]
    if split == "test" or not votes or 2 * sum(votes) == len(votes):
        return None, "missing_or_tied"
    return int(2 * sum(votes) > len(votes)), "crowd_majority"


def mel_bank():
    hz = torch.linspace(0, 8000, 257)
    lo, hi = 2595 * np.log10(1 + 20 / 700), 2595 * np.log10(1 + 8000 / 700)
    edges = 700 * (10 ** (torch.linspace(lo, hi, 66) / 2595) - 1)
    return torch.minimum((hz[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                         (edges[2:, None] - hz[None]) / (edges[2:, None] - edges[1:-1, None])).clamp_min(0)


def encode_audio(path, bank):
    with wave.open(str(path)) as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"unsupported WAV format: {path}")
        rate = stream.getframerate()
        values = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
    values = resample_poly(values, 16000, rate)
    values = np.pad(values[:160000], (0, max(0, 160000 - len(values))))
    x = torch.from_numpy(values)
    spectrum = torch.stft(x, n_fft=512, hop_length=160, win_length=400,
                          window=torch.hann_window(400), return_complex=True).abs().square()
    db = 10 * torch.log10((bank @ spectrum).clamp_min(1e-10))
    image = ((db - db.max()).clamp(-80, 0) + 80) / 80
    image = F.interpolate(image[None, None], (224, 112), mode="bilinear", align_corners=False)[0, 0]
    return normalize_power(image, .5).numpy().astype(np.float16)


def build_text_fields(out):
    words = ["does", "the", "recording", "contain"]
    for event in SONYC_EVENTS:
        words.extend(EVENT_WORDS[event])
    vocab = {"<pad>": 0, "<unk>": 1}
    for word in words:
        vocab.setdefault(word, len(vocab))
    fields = []
    for event in SONYC_EVENTS:
        sentence = ["does", "the", "recording", "contain", *EVENT_WORDS[event]]
        code = torch.zeros(32, 64)
        for position, word in enumerate(sentence):
            code[position, vocab[word]] = 1
        text = F.interpolate(code[None, None], (224, 112), mode="nearest")[0, 0]
        fields.append(normalize_power(text, .5).numpy().astype(np.float16))
    np.save(out / "event_text_fields.npy", np.stack(fields))
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    with args.annotations.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            grouped[row["audio_filename"]].append(row)
    split_names = defaultdict(list)
    for name, rows in grouped.items():
        split = "val" if rows[0]["split"] == "validate" else rows[0]["split"]
        if len({row["split"] for row in rows}) != 1:
            raise ValueError(f"recording occurs in multiple splits: {name}")
        split_names[split].append(name)
    expected = {"train": 13538, "val": 4308, "test": 664}
    if {key: len(value) for key, value in split_names.items()} != expected:
        raise ValueError(f"unexpected annotation identities: { {key: len(value) for key, value in split_names.items()} }")
    all_audio = {}
    for path in args.audio.rglob("*.wav"):
        if path.name in all_audio:
            raise ValueError(f"duplicate audio filename: {path.name}")
        all_audio[path.name] = path
    missing = sorted(set(grouped) - set(all_audio))
    if missing:
        raise ValueError(f"missing {len(missing)} annotated WAV files; first={missing[0]}")
    build_text_fields(args.out)
    bank = mel_bank()
    excluded = Counter()
    query_counts = {}
    for split in ("train", "val", "test"):
        names = sorted(split_names[split])
        audio_fields = np.lib.format.open_memmap(args.out / f"{split}_audio_fields.npy", mode="w+",
                                                 dtype=np.float16, shape=(len(names), 224, 112))
        audio_index, event_index, labels = [], [], []
        for local, name in enumerate(names):
            audio_fields[local] = encode_audio(all_audio[name], bank)
            for event_id, event in enumerate(SONYC_EVENTS):
                label, source = resolved_label(grouped[name], event, split)
                if label is None:
                    excluded[(split, event, source)] += 1
                    continue
                audio_index.append(local); event_index.append(event_id); labels.append(label)
            if (local + 1) % 500 == 0:
                print(json.dumps({"split": split, "audio": local + 1, "queries": len(labels)}), flush=True)
        audio_fields.flush()
        np.save(args.out / f"{split}_audio_index.npy", np.asarray(audio_index, dtype=np.int32))
        np.save(args.out / f"{split}_event_index.npy", np.asarray(event_index, dtype=np.uint8))
        np.save(args.out / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
        query_counts[split] = len(labels)
    protocol = {
        "task": "sonyc", "classes": 2, "dataset": "SONYC-UST v2.3",
        "storage": "sonyc_lazy_v1", "all_original_samples": True,
        "source_recordings": sum(expected.values()), "source_split_recordings": expected,
        "split_queries": query_counts,
        "scope": "all unique recordings listed in the official v2.3 annotations CSV",
        "label_policy": "verified annotator 0 preferred; otherwise crowd majority; ties and unknowns excluded",
        "excluded_queries": {"|".join(key): value for key, value in excluded.items()},
        "events": SONYC_EVENTS,
        "input": "full 10-second 16-kHz log-mel field left, fixed event query right; power 0.5 each",
        "license": "CC BY 4.0", "source": "https://zenodo.org/records/3966543",
        "annotations_sha256": sha256(args.annotations),
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {path.name: sha256(path) for path in args.out.iterdir() if path.is_file()}
    (args.out / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
