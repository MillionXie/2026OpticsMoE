"""CPU-only certification tests; no GPU/model imports or allocation."""
import csv
import json

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import smolvlm2_evaluation_queue as queue


def write(path, obj):
    path.write_text(json.dumps(obj))


def test_training_certification_checks_original_budget_and_artifacts(tmp_path, monkeypatch):
    source = {"entry": "frozen"}
    monkeypatch.setattr(queue.smol, "sources", lambda: source)
    artifacts = {}
    for name in ("best", "last"):
        p = tmp_path / f"{name}_checkpoint.pt"
        p.write_bytes(b"CPU-test-placeholder-not-a-checkpoint")
        artifacts[p.name] = queue.smol.frozen.sha256(p)
        with (tmp_path / f"cpu_{name}_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "target_mos", "prediction_mos"])
            writer.writerows((f"id{i}", 1, 1) for i in range(558))
    receipt = {"status": "complete", "identity": {"source_sha256": source}, "artifacts": artifacts}
    write(tmp_path / "training_receipt.json", receipt)
    write(tmp_path / "status.json", {"status": "complete"})
    write(tmp_path / "train_history.json", [{"epoch": i} for i in range(1, 51)])
    write(tmp_path / "training_report.json", {"backbone_updates": 0, "output_row_parameters": 10240})
    write(tmp_path / "cpu_verification.json", {name: {"strict_cpu_metrics": {"srcc": .7}} for name in ("best", "last")})
    assert queue.verify_training(tmp_path) == receipt
    write(tmp_path / "train_history.json", [{"epoch": i} for i in range(1, 50)])
    with pytest.raises(RuntimeError, match="50 training epochs"):
        queue.verify_training(tmp_path)
    write(tmp_path / "train_history.json", [{"epoch": i} for i in range(1, 51)])
    (tmp_path / "best_checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        queue.verify_training(tmp_path)


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_a100_points_require_their_own_full_predictions(tmp_path, batch):
    report = {"status": "complete", "batch_size_videos": batch, "test_videos": 558,
              "cost_test_videos": 200, "checkpoint_sha256": "same_best",
              "performance": {"srcc": .7}, "cost": {}, "telemetry": {}}
    write(tmp_path / "report.json", report)
    with (tmp_path / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id"])
        writer.writerows((f"id{i}",) for i in range(558))
    assert queue.verify_evaluation(tmp_path, batch, "same_best")["batch"] == batch
    with pytest.raises(RuntimeError, match="mismatched"):
        queue.verify_evaluation(tmp_path, batch, "another_pt")
    with (tmp_path / "predictions.csv").open("a", newline="") as stream:
        csv.writer(stream).writerow(["id0"])
    with pytest.raises(RuntimeError, match="558 unique"):
        queue.verify_evaluation(tmp_path, batch, "same_best")
