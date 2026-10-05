import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
PROFILE = TASK / "configs/lab/rank64_20261002.json"
spec = importlib.util.spec_from_file_location("lab_checkpoint_identity", TASK / "lab_checkpoint_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def test_profile_matches_public_windows_source_identity():
    profile = json.loads(PROFILE.read_text())
    # Fresh clones do not necessarily contain private archive refs. The live
    # source GROUPS comparison is a separate read-only audit, not a unit fixture.
    manifest = json.loads((ROOT / "maintenance/storage/T04_WINDOWS_SOURCE_PRESERVATION_20261005.json").read_text())
    source = next(row for row in manifest["files"] if row["path"] == profile["source_path"])
    assert manifest["source_commit"] == profile["source_commit"]
    assert source["sha256"] == profile["source_sha256"]
    assert set(profile["groups"]) == {"g2", "g5"}
    assert profile["physical_contract"]["exposure_us"] == 2000
    assert profile["physical_contract"]["wait_ms"] == 240


@pytest.mark.parametrize("group", ["g2", "g5"])
def test_exact_architecture_required(group):
    selected = identity.load_identity(PROFILE, group)
    identity.verify_payload_variant({"settings": {"shared_readout_variant": "lowrank64"}}, selected)
    for other in ("lowrank16", "lowrank48", "standard", None):
        with pytest.raises(ValueError):
            identity.verify_payload_variant({"settings": {"shared_readout_variant": other}}, selected)


def test_unaudited_group_rejected():
    with pytest.raises(ValueError):
        identity.load_identity(PROFILE, "g3")


def test_sha_is_checked_before_deserialization(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes, not a torch payload")
    selected = {"filename": checkpoint.name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    assert identity.verify_checkpoint(tmp_path, selected) == checkpoint
    selected["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        identity.verify_checkpoint(tmp_path, selected)


@pytest.mark.parametrize("filename", ["../wrong.pt", "E:/wrong.pt", "dir\\wrong.pt", "wrong.zip"])
def test_unsafe_filename_rejected(tmp_path, filename):
    profile = json.loads(PROFILE.read_text())
    profile["groups"]["g2"]["filename"] = filename
    altered = tmp_path / "bad.json"
    altered.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        identity.load_identity(altered, "g2")
