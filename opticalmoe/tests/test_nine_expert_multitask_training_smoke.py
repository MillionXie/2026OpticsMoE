import importlib.util
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
