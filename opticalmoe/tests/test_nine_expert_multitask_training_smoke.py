import importlib.util
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _load_train_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_nine_expert_as_multitask_moe.py"
    spec = importlib.util.spec_from_file_location("train_nine_expert_as_multitask_moe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyMultitaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, images, task_name=None, **_kwargs):
        return self.net(images.view(images.shape[0], -1))


def test_sequential_backward_multitask_loop_runs_one_update():
    train_script = _load_train_script()
    model = DummyMultitaskModel()
    loaders = {
        "mnist": DataLoader(TensorDataset(torch.rand(2, 1, 2, 2), torch.tensor([0, 1])), batch_size=1),
        "fashionmnist": DataLoader(TensorDataset(torch.rand(2, 1, 2, 2), torch.tensor([1, 2])), batch_size=1),
        "emnist": DataLoader(TensorDataset(torch.rand(2, 1, 2, 2), torch.tensor([2, 0])), batch_size=1),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    step_count = {"value": 0}
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_count["value"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    result = train_script.train_one_epoch_sequential(
        model=model,
        train_loaders=loaders,
        optimizer=optimizer,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(),
        task_names=["mnist", "fashionmnist", "emnist"],
        loss_weights={"mnist": 1.0, "fashionmnist": 1.0, "emnist": 5.0},
        steps_per_epoch=1,
        print_freq=0,
    )

    assert step_count["value"] == 1
    assert result["steps"] == 1
    assert result["samples"] == 3
    assert result["emnist_loss_weight"] == 5.0


def _phase_dropout_config():
    return {
        "layout": {
            "canvas_height": 1000,
            "canvas_width": 1000,
            "input_size": 134,
            "expert_size": 134,
            "expert_pitch": 200,
            "padding": 200,
            "prompt_aperture_size": 600,
        },
        "optics": {
            "num_layers": 1,
            "expert_phase_init": "identity",
            "global_fc_phase_init": "identity",
        },
        "regularization": {
            "phase_dropout": {
                "enabled": True,
                "mode": "block_phase_bypass",
                "expert_p": 0.05,
                "global_fc_p": 0.0,
                "block_size": 8,
                "batch_shared": True,
                "apply_to_experts": True,
                "apply_to_global_fc": False,
                "start_epoch": 10,
            }
        },
        "training": {
            "multitask": {
                "tasks": [
                    {
                        "name": "shape",
                        "head": {"readout_type": "mlp", "hidden_dim": 8},
                    },
                    {
                        "name": "scale",
                        "head": {"readout_type": "mlp", "hidden_dim": 8},
                    },
                ]
            }
        },
    }


def test_nine_expert_phase_dropout_config_is_passed():
    train_script = _load_train_script()
    model = train_script.build_model(
        _phase_dropout_config(),
        task_names=["shape", "scale"],
        task_num_classes={"shape": 3, "scale": 6},
    )

    for layer in model.expert_layers:
        for local_phase in layer.local_phases:
            assert local_phase.phase_dropout_mode == "block_phase_bypass"
            assert local_phase.phase_dropout_p == 0.05
            assert local_phase.phase_dropout_block_size == 8
            assert local_phase.phase_dropout_batch_shared is True
    assert model.global_fc.phase.phase_dropout_mode == "none"
    assert model.global_fc.phase.phase_dropout_p == 0.0


def test_phase_dropout_active_schedule():
    train_script = _load_train_script()
    settings = train_script.phase_dropout_settings(_phase_dropout_config())

    assert train_script.phase_dropout_active_for_epoch(settings, 1) is False
    assert train_script.phase_dropout_active_for_epoch(settings, 9) is False
    assert train_script.phase_dropout_active_for_epoch(settings, 10) is True
    assert train_script.phase_dropout_active_for_epoch(settings, 11) is True


def test_phase_dropout_metrics_written(tmp_path):
    train_script = _load_train_script()
    row = {
        "epoch": 1,
        "phase_dropout_active": False,
        "phase_dropout_mode": "block_phase_bypass",
        "expert_phase_dropout_p": 0.05,
        "global_fc_phase_dropout_p": 0.0,
        "phase_dropout_block_size": 8,
        "phase_dropout_batch_shared": True,
    }
    path = tmp_path / "multitask_metrics.csv"
    train_script.write_rows(path, [row])
    with open(path, newline="", encoding="utf-8") as handle:
        fields = next(csv.DictReader(handle)).keys()

    assert "phase_dropout_active" in fields
    assert "phase_dropout_mode" in fields
    assert "expert_phase_dropout_p" in fields
    assert "global_fc_phase_dropout_p" in fields
    assert "phase_dropout_block_size" in fields
