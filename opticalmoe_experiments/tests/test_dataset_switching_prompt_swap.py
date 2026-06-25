from test_dataset_switching_model import tiny_config

from dataset_switching.scripts.train_dataset_switching import build_model, prompt_swap_evaluation, prompt_swap_summary

import torch
from torch.utils.data import DataLoader, TensorDataset


def test_prompt_swap_matrix_handles_label_space_mismatch():
    task_names = ["mnist", "fashionmnist", "emnist_letters"]
    num_classes = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("learnable_route_moe"), task_names, num_classes)
    loaders = {
        "mnist": DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 1])), batch_size=2),
        "fashionmnist": DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 1])), batch_size=2),
        "emnist_letters": DataLoader(TensorDataset(torch.rand(2, 1, 16, 16), torch.tensor([0, 25])), batch_size=2),
    }
    rows = prompt_swap_evaluation(model, loaders, task_names, num_classes, torch.device("cpu"), torch.nn.CrossEntropyLoss(), max_batches=1)
    assert len(rows) == 27
    mismatch = [row for row in rows if row["eval_dataset"] == "emnist_letters" and row["readout_task"] == "mnist"][0]
    assert mismatch["label_space_matched"] is False
    assert mismatch["accuracy"] == ""
    summary = prompt_swap_summary(rows, task_names)
    assert "mnist" in summary
