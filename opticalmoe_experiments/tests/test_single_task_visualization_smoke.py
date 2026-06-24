import csv
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "single_task" / "visualization"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))

import plot_training_curves


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_training_curves_outputs_all_formats(tmp_path):
    run_dir = tmp_path / "runs" / "run_a"
    _write_csv(
        run_dir / "metrics" / "epoch_metrics.csv",
        [
            {"run_id": "run_a", "epoch": 1, "train_acc": 0.2, "val_acc": 0.1, "train_loss": 2.0, "val_loss": 2.1, "model_type": "lenet5"},
            {"run_id": "run_a", "epoch": 2, "train_acc": 0.4, "val_acc": 0.3, "train_loss": 1.5, "val_loss": 1.7, "model_type": "lenet5"},
        ],
    )
    args = SimpleNamespace(
        run_dirs=[str(run_dir)],
        master_dir=None,
        run_ids=[],
        dataset=None,
        model_type=None,
        model_types=[],
        labels=[],
        out_dir=str(tmp_path / "figures"),
        name="smoke",
        width=5.0,
        height=3.2,
        metric=None,
        metrics=["acc"],
        show=["train", "val"],
        smooth=0,
        mode="overlay",
        show_best=False,
        show_phase_dropout=False,
    )
    plot_training_curves.make_plot(args)
    assert (tmp_path / "figures" / "smoke_accuracy_overlay.png").exists()
    assert (tmp_path / "figures" / "smoke_accuracy_overlay.pdf").exists()
    assert (tmp_path / "figures" / "smoke_accuracy_overlay.svg").exists()
    assert (tmp_path / "figures" / "smoke_accuracy_overlay_plot_data.csv").exists()

