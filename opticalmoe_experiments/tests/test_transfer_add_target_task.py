import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model
from test_dataset_switching_model import tiny_config
from transfer_adaptation.scripts.transfer_utils import add_transfer_target_task


def test_add_transfer_target_task_adds_prompt_head_and_forward():
    tasks = ["mnist", "fashionmnist", "emnist_letters"]
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("learnable_route_moe"), tasks, nums)
    add_transfer_target_task(
        model,
        "usps",
        10,
        init_from_source_prompt="mnist",
        target_head_config={"readout_type": "optical_only", "input_norm": "none"},
    )
    assert "usps" in model.prompt_bank.prompts
    assert "usps" in model.task_readouts
    x = torch.rand(2, 1, 16, 16)
    assert model(x, task_name="usps").shape == (2, 10)

