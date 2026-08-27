from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.lab_validation_package import (
    FRESNEL_GENERATE_COMMAND,
    K1_GENERATE_COMMAND,
    PROJECT_RUNTIME_FILES,
    PackageFile,
    _archive_rows,
    _build_manifest,
    _safe_archive_path,
    _write_zip,
    sha256_file,
    validate_fresnel_v3,
    validate_k1_suite,
    verify_extracted_tree,
    verify_validation_zip,
)


def test_missing_generated_payloads_give_exact_regeneration_commands(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="fresnel_square_aperture_array") as fresnel:
        validate_fresnel_v3(tmp_path / "missing_fresnel")
    assert FRESNEL_GENERATE_COMMAND in str(fresnel.value)

    with pytest.raises(FileNotFoundError, match="00_k1_ready_to_play") as k1:
        validate_k1_suite(tmp_path / "missing_k1")
    assert K1_GENERATE_COMMAND in str(k1.value)


def test_k1_validator_requires_three_strictly_paired_native_bmps(tmp_path: Path) -> None:
    root = tmp_path / "00_k1_ready_to_play"
    pairs = []
    identities = (
        ("checker_c64", "legacy_xy"),
        ("large_blocks_c48_x", "x"),
        ("large_blocks_c48_y", "y"),
    )
    for order, (name, axis) in enumerate(identities, start=1):
        pair_dir = root / f"{order:02d}_{name}"
        pair_dir.mkdir(parents=True)
        amplitude = pair_dir / "amplitude_1024x1024.bmp"
        phase = pair_dir / "phase_1920x1200.bmp"
        preview = pair_dir / "ideal_overlay.png"
        Image.fromarray(np.full((1024, 1024), order * 32, dtype=np.uint8), mode="L").save(
            amplitude, format="BMP"
        )
        Image.fromarray(np.full((1200, 1920), order * 48, dtype=np.uint8), mode="L").save(
            phase, format="BMP"
        )
        Image.fromarray(np.full((16, 16), order * 64, dtype=np.uint8), mode="L").save(
            preview, format="PNG"
        )
        pairs.append(
            {
                "order": order,
                "name": name,
                "grating_axis": axis,
                "k": 1.0,
                "amplitude_bmp": amplitude.relative_to(root).as_posix(),
                "amplitude_sha256": sha256_file(amplitude),
                "phase_bmp": phase.relative_to(root).as_posix(),
                "phase_sha256": sha256_file(phase),
                "preview_png": preview.relative_to(root).as_posix(),
            }
        )
    (root / "README.md").write_text("paired k=1 files\n", encoding="utf-8")
    (root / "k1_pair_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "test",
                "scale_k": 1.0,
                "amplitude_polarity": "255=white/open, 0=black/closed",
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )

    report = validate_k1_suite(root)

    assert report["pairs"] == [name for name, _ in identities]
    assert len(report["files"]) == 11

    pairs[0]["phase_sha256"] = "0" * 64
    (root / "k1_pair_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scale_k": 1.0,
                "amplitude_polarity": "255=white/open, 0=black/closed",
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="phase_sha256 mismatch"):
        validate_k1_suite(root)


def test_validation_zip_and_extracted_tree_are_sha_bound(tmp_path: Path) -> None:
    source_a = tmp_path / "README.md"
    source_b = tmp_path / "tool.py"
    source_a.write_text("laboratory validation\n", encoding="utf-8")
    source_b.write_text("VALUE = 1\n", encoding="utf-8")
    files = [
        PackageFile(source_a, "README_SIM_TO_REAL_LAB.md", "documentation"),
        PackageFile(source_b, "experiments/example/tool.py", "lab_runtime"),
    ]
    rows = _archive_rows(files)
    manifest, checksums = _build_manifest(
        rows=rows,
        fresnel={"manifest_sha256": "1" * 64, "pair_ids": ["pair"]},
        k1={"manifest_sha256": "2" * 64, "pairs": ["checker_c64"]},
        include_vendor_sdk=False,
    )
    archive = tmp_path / "validation.zip"
    _write_zip(archive, files, manifest, checksums)

    verified = verify_validation_zip(archive)

    assert verified["file_count"] == 2
    assert verified["vendor_runtime_included"] is False
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)
    assert verify_extracted_tree(extracted)["status"] == "verified"

    (extracted / "experiments" / "example" / "tool.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="file mismatch"):
        verify_extracted_tree(extracted)


def test_archive_path_guard_and_formal_packager_exclusion() -> None:
    for value in ("../secret", "/absolute", "a/../../secret", "."):
        with pytest.raises(ValueError, match="Unsafe archive path"):
            _safe_archive_path(value)
    assert all(not value.endswith("/lab_package.py") for value in PROJECT_RUNTIME_FILES)
    assert all("lab_bundles" not in value for value in PROJECT_RUNTIME_FILES)
