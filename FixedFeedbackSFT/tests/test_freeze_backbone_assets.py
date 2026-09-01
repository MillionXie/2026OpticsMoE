"""Tests for immutable P11/P13 asset snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from FixedFeedbackSFT.tools import freeze_backbone_assets as freeze


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_manifest(source: Path, content: bytes) -> tuple[dict, list[tuple[Path, str]]]:
    source.write_bytes(content)
    record = {
        "role": "backbone_export",
        "path": "checkpoints/backbone.pt",
        "sha256": digest(content),
        "size_bytes": len(content),
    }
    identity = {
        "format": freeze.SNAPSHOT_FORMAT,
        "label": "teststage",
        "num_stages": 8,
        "config_digest": "a" * 64,
        "stem_checkpoint_sha256": "b" * 64,
        "files": [record],
    }
    manifest = {
        **identity,
        "content_identity_sha256": freeze.canonical_sha256(identity),
        "created_at_utc": "2026-09-02T00:00:00+00:00",
        "source": {},
        "checkpoint_metadata": {},
        "metrics": {},
        "strict_load": {"executed": True, "compatible": True},
    }
    return manifest, [(source, "checkpoints/backbone.pt")]


def test_atomic_snapshot_is_idempotent_and_refuses_different_content(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "runs" / "_assets" / "teststage"
    manifest, sources = make_manifest(source, b"official-backbone")

    assert freeze.commit_snapshot(destination, manifest, sources) == "created"
    assert freeze.commit_snapshot(destination, manifest, sources) == "unchanged"
    assert (destination / "checkpoints" / "backbone.pt").read_bytes() == b"official-backbone"
    assert freeze.verify_existing_snapshot(destination)["content_identity_sha256"] == manifest[
        "content_identity_sha256"
    ]

    changed_manifest, changed_sources = make_manifest(source, b"different-backbone")
    with pytest.raises(freeze.ImmutableSnapshotConflict, match="refusing overwrite"):
        freeze.commit_snapshot(destination, changed_manifest, changed_sources)
    assert (destination / "checkpoints" / "backbone.pt").read_bytes() == b"official-backbone"


def test_snapshot_damage_is_detected_and_never_repaired(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "runs" / "_assets" / "teststage"
    manifest, sources = make_manifest(source, b"official-backbone")
    freeze.commit_snapshot(destination, manifest, sources)
    (destination / "checkpoints" / "backbone.pt").write_bytes(b"tampered")

    with pytest.raises(freeze.ImmutableSnapshotConflict, match="refusing repair"):
        freeze.commit_snapshot(destination, manifest, sources)
    assert (destination / "checkpoints" / "backbone.pt").read_bytes() == b"tampered"


def test_source_policy_requires_a_link_below_fixedfeedback_runs(tmp_path: Path) -> None:
    fixed = tmp_path / "2026OpticsMoE" / "FixedFeedbackSFT"
    runs = fixed / "runs"
    physical = runs / "project" / "run"
    physical.mkdir(parents=True)
    with pytest.raises(freeze.SnapshotError, match="registered symlink"):
        freeze._require_registered_link(physical, runs, kind="test run")
    with pytest.raises(freeze.SnapshotError, match="below the central runs root"):
        freeze._require_registered_link(tmp_path / "outside", runs, kind="test run")


def test_manifest_sidecar_covers_the_exact_manifest_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "runs" / "_assets" / "teststage"
    manifest, sources = make_manifest(source, b"official-backbone")
    freeze.commit_snapshot(destination, manifest, sources)

    manifest_bytes = (destination / "manifest.json").read_bytes()
    expected = hashlib.sha256(manifest_bytes).hexdigest()
    sidecar = (destination / "manifest.sha256").read_text(encoding="ascii").strip()
    assert sidecar == f"{expected}  manifest.json"
    assert json.loads(manifest_bytes)["format"] == freeze.SNAPSHOT_FORMAT


def test_identity_guard_requires_declared_matching_depth() -> None:
    spec = freeze.StageSpec(
        label="teststage",
        num_stages=8,
        project="project",
        run_name="run",
        config_digest="a" * 64,
        backbone_sha256="b" * 64,
        training_sha256="c" * 64,
        assets=(),
    )
    records = [
        {"role": "backbone_export", "sha256": "b" * 64},
        {"role": "best_training_checkpoint", "sha256": "c" * 64},
        {"role": "frozen_qwen_stem", "sha256": freeze.OFFICIAL_STEM_SHA256},
    ]
    metadata = {
        "backbone_export": {
            "config_digest": "a" * 64,
            "num_stages": 8,
            "stem_checkpoint_sha256": freeze.OFFICIAL_STEM_SHA256,
        },
        "best_training_checkpoint": {
            "config_digest": "a" * 64,
            "num_stages": None,
            "stem_checkpoint_sha256": None,
        },
    }
    freeze._checkpoint_identity_guard(spec, records, metadata)
    metadata["backbone_export"]["num_stages"] = 16
    with pytest.raises(freeze.SnapshotError, match="declares depth 16"):
        freeze._checkpoint_identity_guard(spec, records, metadata)


def test_command_defaults_remain_inside_fixedfeedback_runs() -> None:
    args = freeze.parse_args([])
    runs = Path(os.path.abspath(args.runs_root))
    assert runs.name == "runs"
    assert runs.parent.name == "FixedFeedbackSFT"
    assert Path(os.path.abspath(args.eight_run)).is_relative_to(runs)
    assert Path(os.path.abspath(args.sixteen_run)).is_relative_to(runs)
    assert Path(os.path.abspath(args.stem_link)).is_relative_to(runs)
    assert freeze.ASSET_DIRECTORY_NAME == "_assets"


def test_freezer_rejects_same_named_runs_tree_outside_repository(tmp_path: Path) -> None:
    outside = tmp_path / "FixedFeedbackSFT" / "runs"
    outside.mkdir(parents=True)
    with pytest.raises(freeze.SnapshotError, match="current repository's exact"):
        freeze.freeze_official_assets(
            outside,
            outside / "p11",
            outside / "p13",
            outside / "stem.pt",
        )
