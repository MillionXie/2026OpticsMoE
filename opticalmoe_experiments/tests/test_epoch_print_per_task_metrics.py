import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_switching.scripts.train_dataset_switching as ds_train
import same_input_multitask.scripts.train_same_input_multitask as simt_train
from test_dataset_switching_model import tiny_config
from test_dataset_switching_training_smoke import Args as DSArgs, _fake_loaders as ds_fake_loaders
from test_same_input_multitask_training_smoke import Args as SIMTArgs, _config as simt_config, _fake_loaders as simt_fake_loaders


def test_dataset_switching_epoch_prints_per_task_metrics(monkeypatch, tmp_path, capsys):
    cfg = tiny_config("learnable_route_moe")
    cfg["experiment"] = {"run_name": "dataset_switching_print"}
    cfg["training"]["epochs"] = 1
    cfg["training"]["multitask"].update({"steps_per_epoch": 1, "loss_weights": {"mnist": 1.0, "fashionmnist": 1.0, "emnist_letters": 1.0}})
    cfg["training"]["evaluation"] = {"max_val_batches": 1, "max_test_batches": 1}
    cfg["visualization"] = {"enabled": False}
    cfg["optimizer"] = {"type": "adamw", "lr": 0.001, "weight_decay": 0.0}
    monkeypatch.setattr(ds_train, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(ds_train, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ds_train, "create_task_loaders", ds_fake_loaders)
    ds_train.run_training(cfg, DSArgs())
    out = capsys.readouterr().out
    assert "mnist" in out and "train_loss=" in out and "val_acc=" in out
    assert "fashionmnist" in out and "emnist_letters" in out


def test_same_input_epoch_prints_per_task_metrics(monkeypatch, tmp_path, capsys):
    cfg = simt_config()
    monkeypatch.setattr(simt_train, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(simt_train, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(simt_train, "create_same_input_multitask_dataloaders", simt_fake_loaders)
    simt_train.run_training(cfg, SIMTArgs())
    out = capsys.readouterr().out
    assert "shape" in out and "scale" in out
    assert "train_loss=" in out and "val_acc=" in out
