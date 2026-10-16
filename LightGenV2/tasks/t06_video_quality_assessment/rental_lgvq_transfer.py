"""Exact fixed-split DATA-only export/import for the authorized rental host."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

MANIFEST_SHA = "607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc"
SOURCE = Path("/DATA/DATA1/guest3/2026OpticsMoE/experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv")
DESTINATION = Path("/data/lgvq")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def add_bytes(archive, name, value):
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))

def export():
    if sha(SOURCE) != MANIFEST_SHA:
        raise RuntimeError("Original split SHA changed")
    rows = list(csv.DictReader(SOURCE.open()))
    if len(rows) != 2808 or len({r['sample_id'] for r in rows}) != 2808:
        raise RuntimeError("Original 2808 unique samples required")
    manifest = []
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        add_bytes(archive, "original_manifest.csv", SOURCE.read_bytes())
        for i, row in enumerate(rows):
            source = Path(row["video_path"])
            name = f"videos/{i:05d}{source.suffix}"
            manifest.append(dict(path=name, sha256=sha(source), bytes=source.stat().st_size,
                                 sample_id=row["sample_id"]))
            archive.add(source, arcname=name, recursive=False)
            row["video_path"] = str(DESTINATION / name)
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
        add_bytes(archive, "manifest.csv", text.getvalue().encode())
        add_bytes(archive, "data_transfer_manifest.json", json.dumps(dict(original_manifest_sha256=MANIFEST_SHA,
                    samples=2808, files=manifest), indent=2).encode())

def receive(path, expected):
    if sha(path) != expected:
        raise RuntimeError("Transport SHA mismatch")
    DESTINATION.mkdir(exist_ok=False)
    with tarfile.open(path) as archive:
        for member in archive:
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or not member.isfile():
                raise RuntimeError("Unsafe DATA archive member")
            target = DESTINATION / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
    payload = json.loads((DESTINATION / "data_transfer_manifest.json").read_text())
    if sha(DESTINATION / "original_manifest.csv") != MANIFEST_SHA or len(payload["files"]) != 2808:
        raise RuntimeError("Wrong original data identity")
    for record in payload["files"]:
        item = DESTINATION / record["path"]
        if item.stat().st_size != record["bytes"] or sha(item) != record["sha256"]:
            raise RuntimeError("Transferred video SHA mismatch")
    original = list(csv.DictReader((DESTINATION / "original_manifest.csv").open()))
    remapped = list(csv.DictReader((DESTINATION / "manifest.csv").open()))
    for a, b in zip(original, remapped):
        if {k:v for k,v in a.items() if k != "video_path"} != {k:v for k,v in b.items() if k != "video_path"}:
            raise RuntimeError("Only absolute video paths may change")
    result = dict(status="verified", original_manifest_sha256=MANIFEST_SHA,
                  remapped_manifest_sha256=sha(DESTINATION / "manifest.csv"), videos=2808,
                  video_bytes=sum(r["bytes"] for r in payload["files"]), transport_sha256=expected)
    (DESTINATION / "transfer_receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("export", "import"), required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.phase == "export":
        export()
    else:
        if args.archive.resolve() != Path("/data/lgvq_transfer.tar.partial") or len(args.sha256 or "") != 64:
            raise ValueError("Only the explicit rental DATA archive is allowed")
        receive(args.archive, args.sha256)

if __name__ == "__main__":
    main()
