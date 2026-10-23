import json
from pathlib import Path

import pytest

from LightGenV2.tasks.t19_resnet20_baselines.inspect_preparation import (
    TASK_DIR, inspect, save_new_report,
)


def test_exact_nine_sources_exist_and_training_is_still_blocked():
    report = inspect()
    assert len(report["models"]) == 9
    assert len(report["baseline_source_files"]) >= 30
    assert not report["cuda_initialized"]
    assert not report["formal_training_started"]
    assert not report["real_dataset_adapters_implemented"]
    assert report["execution_authorization"]["training_authorized_after_required_preflight"]
    assert "user_accepts_architecture_and_scope" not in report["pending_gates"]
    config = json.loads((TASK_DIR / "configs/suite_proposal.json").read_text(encoding="utf-8"))
    assert config["resource_policy"]["max_gpus"] == 3
    assert not config["training_authorized"]
    assert not config["architecture_approved"]


def test_no_output_outside_task_reports(tmp_path):
    with pytest.raises(ValueError, match="must stay"):
        save_new_report(tmp_path / "report.json", {})


def test_inspection_cli_cannot_launch_training():
    source = (TASK_DIR / "inspect_preparation.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "--train" not in source
