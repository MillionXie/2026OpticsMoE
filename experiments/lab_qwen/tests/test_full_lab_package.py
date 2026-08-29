from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.lab_qwen.full_lab_package import (
    CALTECH101_CATEGORIES,
    _add_caltech101_local_finetune_data,
    _add_ready_bmps_from_compact,
    _require_session_checkpoint,
)


def test_ready_bmp_packaging_includes_bound_reconstruction_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session" / "04_language_global" / "compact_amplitude"
    source.mkdir(parents=True)
    compact = (np.arange(478 * 478, dtype=np.uint32) % 256).astype(np.uint8)
    compact = compact.reshape(478, 478)
    compact_path = source / "probe.png"
    Image.fromarray(compact, mode="L").save(compact_path)
    entries = {}

    count = _add_ready_bmps_from_compact(
        entries,
        source_session=tmp_path / "session",
        destination="experiments/lab_qwen/agree",
        category="test",
    )

    assert count == 1
    root = "experiments/lab_qwen/agree/04_language_global/amplitude_to_play"
    output_data = entries[f"{root}/probe.bmp"].read()
    manifest_data = entries[f"{root}/reconstruction_manifest.csv"].read()
    rows = list(csv.DictReader(io.StringIO(manifest_data.decode("utf-8-sig"))))
    assert len(rows) == 1
    assert rows[0]["source_png"] == "probe.png"
    assert rows[0]["output_bmp"] == "probe.bmp"
    assert rows[0]["source_sha256"] == hashlib.sha256(
        compact_path.read_bytes()
    ).hexdigest()
    assert rows[0]["output_sha256"] == hashlib.sha256(output_data).hexdigest()
    assert rows[0]["active_bounds_xyxy"] == "273,273,751,751"

    with Image.open(io.BytesIO(output_data)) as image:
        assert image.mode == "L"
        assert image.size == (1024, 1024)
        reconstructed = np.asarray(image)
    assert np.array_equal(reconstructed[273:751, 273:751], compact)
    assert int(reconstructed[:273].sum()) == 0


def test_four_stage_checkpoint_audit_follows_sequential_chain(tmp_path: Path) -> None:
    session = tmp_path / "four"
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"initial")
    expected = hashlib.sha256(initial.read_bytes()).hexdigest()
    stages = (
        ("01_vision_expert", expected),
        ("02_vision_global", hashlib.sha256(b"after-v1").hexdigest()),
        ("03_language_expert", hashlib.sha256(b"after-v2").hexdigest()),
        ("04_language_global", hashlib.sha256(b"after-l1").hexdigest()),
    )
    checkpoints = session / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "after_vision_expert.pt").write_bytes(b"after-v1")
    (checkpoints / "after_vision_global.pt").write_bytes(b"after-v2")
    (checkpoints / "after_language_expert.pt").write_bytes(b"after-l1")
    for stage, digest in stages:
        directory = session / stage
        directory.mkdir()
        (directory / "transport_spec.json").write_text(
            '{"checkpoint_sha256":"' + digest + '"}', encoding="utf-8"
        )
    _require_session_checkpoint(session, expected, sequential_four_stage=True)

    drifted = session / "03_language_expert" / "transport_spec.json"
    drifted.write_text('{"checkpoint_sha256":"' + "0" * 64 + '"}', encoding="utf-8")
    import pytest

    with pytest.raises(RuntimeError, match="different checkpoint"):
        _require_session_checkpoint(session, expected, sequential_four_stage=True)


def test_local_finetune_package_embeds_complete_target10_categories(
    tmp_path: Path,
) -> None:
    categories = tmp_path / "data/Caltech101/caltech-101/101_ObjectCategories"
    for category in CALTECH101_CATEGORIES:
        directory = categories / category
        directory.mkdir(parents=True)
        (directory / "image_0001.jpg").write_bytes(category.encode("ascii"))
        (directory / "image_0002.jpg").write_bytes(category.encode("ascii") + b"2")
    entries = {}

    counts = _add_caltech101_local_finetune_data(entries, tmp_path)

    assert counts == {category: 2 for category in CALTECH101_CATEGORIES}
    assert "data/Caltech101/BUNDLED_TARGET10.json" in entries
    for category in CALTECH101_CATEGORIES:
        prefix = f"data/Caltech101/101_ObjectCategories/{category}/"
        assert sum(name.startswith(prefix) for name in entries) == 2
