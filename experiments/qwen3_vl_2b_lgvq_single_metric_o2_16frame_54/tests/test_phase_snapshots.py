from __future__ import annotations

from pathlib import Path

from ..modeling import build_model
from ..phase_snapshots import load_phase_snapshot, save_phase_snapshot, summarize_directory
from .test_model_and_training import _small_settings


def test_phase_snapshot_is_phase_only_and_strictly_readable(tmp_path: Path) -> None:
    settings = _small_settings(tmp_path)
    model = build_model(settings)
    path = save_phase_snapshot(model, settings, epoch=5, metrics={"srcc": 0.75})
    payload = load_phase_snapshot(path)

    assert path.name == "epoch_0005.pt"
    assert payload["epoch"] == 5
    assert payload["target_name"] == "spatial"
    assert len(payload["planes"]) == 6
    assert "state_dict" not in payload
    assert "optimizer" not in payload
    assert payload["unmodulated_leakage"]["train_min"] >= 0.20
    assert set(payload["fusion_alpha"]) == {
        "vision_expert",
        "vision_global",
        "language_expert",
        "language_global",
    }

    report = summarize_directory(path.parent)
    assert report["epochs"] == [5]
    assert (path.parent / "phase_evolution_summary.csv").is_file()
