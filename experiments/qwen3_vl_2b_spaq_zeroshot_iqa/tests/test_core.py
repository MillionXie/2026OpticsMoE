from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.qwen3_vl_2b_spaq_zeroshot_iqa.generation import (
    parse_score,
    zeroshot_metrics,
)
from experiments.qwen3_vl_2b_spaq_zeroshot_iqa.run import PHASES, main
from experiments.qwen3_vl_2b_spaq_zeroshot_iqa.settings import Settings, load_settings


def _make_spaq(root: Path, count: int = 20) -> None:
    image_dir = root / "TestImage"
    image_dir.mkdir(parents=True)
    rows = []
    for index in range(count):
        name = f"image_{index:03d}.png"
        Image.new("RGB", (12, 10), color=(index, index, index)).save(image_dir / name)
        rows.append(
            {
                "Image name": name,
                "MOS": 20 + index,
                "Brightness": 30 + index,
                "Colorfulness": 40 + index,
                "Contrast": 50 + index,
            }
        )
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("75", 75.0),
        ("75.5%", 75.5),
        ("Score: 62", 62.0),
        ("The rating is 41.25.", 41.25),
        ("75/100", 75.0),
    ],
)
def test_parse_valid_scores(text: str, expected: float) -> None:
    value, _ = parse_score(text)
    assert value == expected


@pytest.mark.parametrize("text", ["no score", "-1", "101", "nan"])
def test_parse_failures_are_not_clipped(text: str) -> None:
    value, _ = parse_score(text)
    assert value is None


def test_zeroshot_metrics_and_tolerance_accuracy() -> None:
    rows = []
    for task in ("MOS", "Brightness", "Colorfulness", "Contrast"):
        rows.extend(
            [
                {"task": task, "image_name": "a", "true_score": 10, "predicted_score": 10, "parse_valid": True},
                {"task": task, "image_name": "b", "true_score": 20, "predicted_score": 25, "parse_valid": True},
                {"task": task, "image_name": "c", "true_score": 30, "predicted_score": None, "parse_valid": False},
            ]
        )
    metrics = zeroshot_metrics(rows)
    assert metrics["parse_rate"] == pytest.approx(2 / 3)
    assert metrics["macro_average"]["mae"] == pytest.approx(2.5)
    assert metrics["macro_average"]["within_5_accuracy"] == 1.0


def test_config_and_no_training_phase(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "data_root": str(tmp_path / "SPAQ"),
                "output_dir": str(tmp_path / "run"),
                "device": "cpu",
                "dtype": "float32",
                "attn_implementation": "eager",
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.generation_batch_size == 4
    assert "train" not in PHASES
    assert "evaluate" in PHASES


def test_prepare_data_cli(tmp_path: Path) -> None:
    data_root = tmp_path / "SPAQ"
    _make_spaq(data_root, 10)
    output_dir = tmp_path / "run"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "device": "cpu",
                "dtype": "float32",
                "attn_implementation": "eager",
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config), "--phase", "prepare_data"]) == 0
    assert (output_dir / "data_split.json").is_file()
    assert (output_dir / "dataset.json").is_file()


def test_settings_reject_invalid_generation_values() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        Settings(max_new_tokens=0).validate()

