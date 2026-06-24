import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SINGLE = ROOT / "single_task"
if str(SINGLE) not in sys.path:
    sys.path.insert(0, str(SINGLE))

from common.reporting.metrics_writer import write_rows
from common.training.checkpointing import save_checkpoint
from common.training.train_loop import train_one_epoch
from common.utils.config import save_json, save_yaml
from baselines.model_factory import build_model, build_optimizer
from test_single_task_models import tiny_config


def test_training_smoke_outputs(tmp_path):
    cfg = tiny_config("learnable_route_moe")
    cfg["optimizer"] = {"type": "adamw", "lr": 0.001, "weight_decay": 0.0}
    model = build_model(cfg, num_classes=10)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.rand(2, 1, 32, 32), torch.tensor([0, 1])),
        batch_size=1,
    )
    optimizer = build_optimizer(model, cfg)
    metrics = train_one_epoch(model, loader, torch.nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), print_freq=0)
    run_dir = tmp_path / "run"
    save_yaml(cfg, run_dir / "config.yaml")
    save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, 1, metrics, cfg)
    write_rows(run_dir / "metrics" / "epoch_metrics.csv", [{"epoch": 1, **metrics}])
    save_json({"run_id": "smoke"}, run_dir / "summary.json")
    save_json({"run_id": "smoke"}, run_dir / "summary_for_master" / "runs_rows.json")
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "metrics" / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary_for_master" / "runs_rows.json").exists()

