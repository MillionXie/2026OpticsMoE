from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.lab_qwen.full_lab_package import _add_ready_bmps_from_compact


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
