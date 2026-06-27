import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model
from test_dataset_switching_model import tiny_config
from transfer_adaptation.scripts.transfer_utils import add_transfer_target_task, target_prompt_swap_rows


def test_transfer_prompt_swap_uses_target_readout_for_all_prompts():
    source_tasks = ["mnist", "fashionmnist", "emnist_letters"]
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("learnable_route_moe"), source_tasks, nums)
    add_transfer_target_task(model, "usps", 10, target_head_config={"readout_type": "optical_only", "input_norm": "none"})
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 1])), batch_size=2)
    rows, summary = target_prompt_swap_rows(
        model,
        loader,
        source_tasks,
        "usps",
        "usps",
        torch.device("cpu"),
        torch.nn.CrossEntropyLoss(),
        "run",
        max_batches=1,
    )
    assert {row["prompt_task"] for row in rows} == {"usps", "mnist", "fashionmnist", "emnist_letters"}
    assert {row["readout_task"] for row in rows} == {"usps"}
    assert "target_prompt_gap" in summary

