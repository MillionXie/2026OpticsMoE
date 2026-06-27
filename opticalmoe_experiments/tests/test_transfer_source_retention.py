import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model
from test_dataset_switching_model import tiny_config
from transfer_adaptation.scripts.transfer_utils import (
    add_transfer_target_task,
    evaluate_source_tasks,
    source_retention_rows,
)


def test_source_retention_is_unchanged_when_only_target_prompt_changes():
    source_tasks = ["mnist", "fashionmnist", "emnist_letters"]
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("learnable_route_moe"), source_tasks, nums)
    add_transfer_target_task(model, "usps", 10, target_head_config={"readout_type": "optical_only", "input_norm": "none"})
    loaders = {
        "mnist": DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 1])), batch_size=2),
        "fashionmnist": DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 1])), batch_size=2),
        "emnist_letters": DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 25])), batch_size=2),
    }
    before = evaluate_source_tasks(model, loaders, source_tasks, torch.device("cpu"), torch.nn.CrossEntropyLoss(), max_batches=1)
    with torch.no_grad():
        model.prompt_bank.prompts["usps"].amplitude_logits.add_(0.5)
    after = evaluate_source_tasks(model, loaders, source_tasks, torch.device("cpu"), torch.nn.CrossEntropyLoss(), max_batches=1)
    rows, summary = source_retention_rows("run", before, after, source_tasks)
    assert {row["source_task"] for row in rows} == set(source_tasks)
    assert abs(summary["max_source_acc_drop"]) < 1e-12

