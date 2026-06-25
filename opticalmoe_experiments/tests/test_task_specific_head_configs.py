import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model as build_dataset_model
from same_input_multitask.scripts.train_same_input_multitask import build_model as build_same_input_model
from test_dataset_switching_model import tiny_config


def test_dataset_switching_resolves_per_task_heads():
    cfg = tiny_config("learnable_route_moe")
    tasks = ["mnist", "fashionmnist", "emnist_letters"]
    cfg["training"]["multitask"]["tasks"][0]["head"] = {"hidden_dim": 64, "activation": "relu"}
    cfg["training"]["multitask"]["tasks"][1]["head"] = {"hidden_dim": 64, "activation": "gelu"}
    cfg["training"]["multitask"]["tasks"][2]["head"] = {"hidden_dim": 96, "activation": "gelu"}
    model = build_dataset_model(cfg, tasks, {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26})

    assert model.task_head_configs["mnist"]["hidden_dim"] == 64
    assert model.task_head_configs["fashionmnist"]["hidden_dim"] == 64
    assert model.task_head_configs["emnist_letters"]["hidden_dim"] == 96
    assert model.task_readouts["mnist"] is not model.task_readouts["fashionmnist"]
    counts = model.task_readout_parameter_counts()
    assert set(counts) == set(tasks)
    assert all(value > 0 for value in counts.values())
    assert model.electronic_parameter_count() == sum(counts.values())


def test_same_input_resolves_training_task_heads():
    cfg = tiny_config("learnable_route_moe")
    cfg["training"] = {
        "tasks": ["shape", "scale", "x_position_4bin"],
        "task_heads": {
            "shape": {"hidden_dim": 32, "activation": "relu"},
            "scale": {"hidden_dim": 64, "activation": "gelu"},
            "x_position_4bin": {"hidden_dim": 48, "activation": "relu"},
        },
    }
    model = build_same_input_model(
        cfg,
        ["shape", "scale", "x_position_4bin"],
        {"shape": 3, "scale": 6, "x_position_4bin": 4},
    )
    assert model.task_head_configs["shape"]["hidden_dim"] == 32
    assert model.task_head_configs["scale"]["hidden_dim"] == 64
    assert model.task_head_configs["x_position_4bin"]["hidden_dim"] == 48
    assert model.task_readouts["shape"] is not model.task_readouts["scale"]
