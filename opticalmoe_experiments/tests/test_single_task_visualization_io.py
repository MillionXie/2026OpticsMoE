import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "single_task" / "visualization"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))

from common.utils.config import save_json

spec = importlib.util.spec_from_file_location("single_task_visualization_io", VIS_DIR / "io.py")
viz_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz_io)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_visualization_io_loads_run_and_resolves_master(tmp_path):
    run_dir = tmp_path / "runs" / "run_a"
    _write_csv(
        run_dir / "metrics" / "epoch_metrics.csv",
        [{"run_id": "run_a", "epoch": 1, "train_acc": 0.1, "val_acc": 0.2}],
    )
    save_json({"run_id": "run_a", "final_test_acc": 0.3}, run_dir / "metrics" / "final_metrics.json")
    save_json({"run_id": "run_a", "dataset_name": "mnist", "model_type": "lenet5"}, run_dir / "summary.json")

    epoch_rows = viz_io.load_run_epoch_metrics(run_dir)
    final = viz_io.load_run_final_metrics(run_dir)
    assert epoch_rows[0]["run_id"] == "run_a"
    assert final["dataset_name"] == "mnist"
    assert final["model_type"] == "lenet5"

    _write_csv(tmp_path / "results" / "master_runs.csv", [{"run_id": "run_a", "run_dir": str(run_dir)}])
    args = SimpleNamespace(run_dirs=[], master_dir=str(tmp_path / "results"), run_ids=["run_a"])
    resolved = viz_io.resolve_runs_from_args(args)
    assert resolved == [run_dir]

