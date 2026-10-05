"""Validate an explicit lab checkpoint identity without importing models or SDKs.

The config is not an instruction to replace a live checkpoint or switch a
historical capture entry. Callers must still use the matching model variant.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def load_identity(path: Path, group: str) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported lab identity schema")
    variant = profile.get("shared_readout_variant")
    if variant != "lowrank64":
        raise ValueError("This identity profile is audited only for lowrank64")
    row = profile.get("groups", {}).get(group)
    if not isinstance(row, dict):
        raise ValueError("Group is not audited for this capacity")
    filename = row.get("filename", "")
    if (not filename or Path(filename).name != filename or "/" in filename
            or "\\" in filename or ":" in filename or not filename.endswith(".pt")):
        raise ValueError("Checkpoint filename must be a plain PT basename")
    if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
        raise ValueError("Missing exact checkpoint SHA256")
    return {"group": group, "shared_readout_variant": variant, **row}


def verify_checkpoint(weights: Path, identity: dict) -> Path:
    """Read and hash only the explicitly selected PT; never deserialize it."""
    checkpoint = Path(weights) / identity["filename"]
    with checkpoint.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != identity["sha256"]:
        raise ValueError("Checkpoint SHA mismatch; do not substitute a nearby version")
    return checkpoint


def verify_payload_variant(payload: dict, identity: dict) -> None:
    actual = payload.get("settings", {}).get("shared_readout_variant")
    if actual != identity["shared_readout_variant"]:
        raise ValueError("Checkpoint architecture does not match the explicit lab identity")
