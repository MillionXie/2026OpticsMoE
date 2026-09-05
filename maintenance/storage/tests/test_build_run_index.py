from __future__ import annotations

from pathlib import Path

from maintenance.storage.build_run_index import build_rows, find_run_roots


def test_central_fixed_feedback_runs_are_indexed_per_project(tmp_path: Path) -> None:
    project = "example_fa_project"
    source = tmp_path / "FixedFeedbackSFT" / "projects" / project
    source.mkdir(parents=True)
    run = tmp_path / "FixedFeedbackSFT" / "runs" / project / "formal_run"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    logs = tmp_path / "FixedFeedbackSFT" / "runs" / project / "logs"
    logs.mkdir()
    (logs / "train.log").write_text("ignored", encoding="utf-8")

    roots = find_run_roots(tmp_path)
    assert roots == [tmp_path / "FixedFeedbackSFT" / "runs" / project]

    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["owner"] == f"FixedFeedbackSFT/projects/{project}"
    assert rows[0]["runs_root"] == f"FixedFeedbackSFT/runs/{project}"
    assert rows[0]["run_name"] == "formal_run"
    assert rows[0]["run_path"] == f"FixedFeedbackSFT/runs/{project}/formal_run"


def test_legacy_project_local_runs_keep_existing_behavior(tmp_path: Path) -> None:
    run = tmp_path / "experiments" / "ordinary_project" / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"checkpoint")

    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["owner"] == "experiments/ordinary_project"
    assert rows[0]["run_path"] == "experiments/ordinary_project/runs/trial"
