import csv
import math
import sys
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for search_path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from opticalmoe.optics.four_expert_multitask_moe import (  # noqa: E402
    FourExpertMultitaskMoEClassifier,
    TaskPromptBank,
)
from opticalmoe.training.four_expert_reporting import (  # noqa: E402
    build_architecture_report,
    save_initial_state,
)
from opticalmoe.training.multitask_engine import (  # noqa: E402
    task_switching_evaluation,
    train_multitask_one_epoch,
)
import train_four_expert_moe_v2 as single_script  # noqa: E402
import train_four_expert_multitask_moe as multitask_script  # noqa: E402


class _DummyPhaseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(4, 8, 8))

    def get_phase_wrapped(self):
        return torch.remainder(self.phase, 2.0 * math.pi)


class _ReportModel(nn.Module):
    num_classes = 10

    def __init__(self):
        super().__init__()
        self.optical = nn.Parameter(torch.zeros(5))
        self.prompt = nn.Parameter(torch.zeros(4))

    def optical_parameter_count(self):
        return 9

    def prompt_parameter_count(self):
        return 4

    def electronic_parameter_count(self):
        return 0


class _InitialStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 5
        self.expert_layers = nn.ModuleList(
            [_DummyPhaseLayer() for _ in range(self.num_layers)]
        )


class _TinyMultitaskModel(nn.Module):
    """Small proxy used to test the task-aware optimizer loop without FFTs."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 10)
        self.task_bias = nn.ParameterDict(
            {
                "mnist": nn.Parameter(torch.zeros(10)),
                "fashionmnist": nn.Parameter(torch.ones(10) * 0.01),
                "emnist": nn.Parameter(torch.ones(10) * 0.02),
            }
        )

    def forward(self, images, task_name, return_intermediates=False):
        logits = self.backbone(images.view(images.shape[0], -1))
        logits = logits + self.task_bias[task_name]
        if return_intermediates:
            return logits, {"task_name": task_name}
        return logits


def _tiny_loaders():
    images = torch.arange(32, dtype=torch.float32).view(8, 1, 2, 2) / 32.0
    labels = torch.arange(8, dtype=torch.long) % 10
    mnist = DataLoader(TensorDataset(images, labels), batch_size=2)
    fashion = DataLoader(
        TensorDataset(images.flip(0), labels.flip(0)),
        batch_size=2,
    )
    emnist = DataLoader(
        TensorDataset(torch.roll(images, shifts=1, dims=0), labels),
        batch_size=2,
    )
    return {"mnist": mnist, "fashionmnist": fashion, "emnist": emnist}


def test_task_prompt_bank_keeps_tasks_independent():
    bank = TaskPromptBank(["mnist", "fashionmnist", "emnist"])
    assert bank.amplitudes("mnist").shape == (4,)
    assert bank.amplitudes("fashionmnist").shape == (4,)
    assert bank.amplitudes("emnist").shape == (4,)
    assert (
        bank.amplitude_logits["mnist"]
        is not bank.amplitude_logits["fashionmnist"]
    )
    with torch.no_grad():
        bank.amplitude_logits["mnist"][0] = -2.0
    assert not torch.allclose(
        bank.amplitudes("mnist"),
        bank.amplitudes("fashionmnist"),
    )


def test_multitask_optical_forward_accepts_three_task_names():
    model = FourExpertMultitaskMoEClassifier(
        task_names=["mnist", "fashionmnist", "emnist"],
        num_layers=1,
        expert_phase_init="identity",
        global_fc_phase_init="identity",
    )
    images = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        mnist_logits = model(images, task_name="mnist")
        fashion_logits = model(images, task_name="fashionmnist")
        emnist_logits = model(images, task_name="emnist")
        task_id_logits = model(images, task_id=0)
    assert mnist_logits.shape == (1, 10)
    assert fashion_logits.shape == (1, 10)
    assert emnist_logits.shape == (1, 10)
    assert task_id_logits.shape == (1, 10)


def test_multitask_training_loop_runs_one_tiny_epoch():
    model = _TinyMultitaskModel()
    loaders = _tiny_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    result = train_multitask_one_epoch(
        model=model,
        train_loaders=loaders,
        optimizer=optimizer,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(),
        task_names=["mnist", "fashionmnist"],
    )
    assert result["steps"] == 4
    assert result["total_loss"] > 0.0
    assert result["joint_sample_loss"] > 0.0
    assert 0.0 <= result["joint_accuracy"] <= 1.0
    assert result["samples"] == 16
    assert "mnist_acc" in result
    assert "fashionmnist_acc" in result


def test_multitask_training_loop_runs_three_task_epoch():
    model = _TinyMultitaskModel()
    loaders = _tiny_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    result = train_multitask_one_epoch(
        model=model,
        train_loaders=loaders,
        optimizer=optimizer,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(),
        task_names=["mnist", "fashionmnist", "emnist"],
        steps_per_epoch=2,
        print_freq=0,
    )
    assert result["steps"] == 2
    assert result["samples"] == 12
    assert result["mnist_samples"] == 4
    assert result["fashionmnist_samples"] == 4
    assert result["emnist_samples"] == 4
    assert "emnist_acc" in result


def test_multitask_training_loop_supports_fixed_step_budget():
    model = _TinyMultitaskModel()
    loaders = _tiny_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    result = train_multitask_one_epoch(
        model=model,
        train_loaders=loaders,
        optimizer=optimizer,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(),
        task_names=["mnist", "fashionmnist"],
        steps_per_epoch=2,
        print_freq=0,
    )
    assert result["steps"] == 2
    assert result["available_steps"] == 4
    assert result["mnist_samples"] == 4
    assert result["fashionmnist_samples"] == 4


def test_task_switching_results_can_be_written(tmp_path):
    model = _TinyMultitaskModel()
    loaders = _tiny_loaders()
    rows = task_switching_evaluation(
        model=model,
        test_loaders=loaders,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(),
        task_names=["mnist", "fashionmnist", "emnist"],
    )
    output = tmp_path / "task_switching_eval.csv"
    multitask_script.write_rows(output, rows)
    with open(output, newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert output.exists()
    assert len(saved) == 9
    assert {row["eval_dataset"] for row in saved} == {
        "mnist",
        "fashionmnist",
        "emnist",
    }
    assert {row["prompt_task"] for row in saved} == {
        "mnist",
        "fashionmnist",
        "emnist",
    }


def test_optimizer_builder_uses_adamw_003():
    model = nn.Linear(2, 2)
    config = {
        "optimizer": {
            "type": "adamw",
            "lr": 0.003,
            "weight_decay": 0.0,
        }
    }
    optimizer = single_script.optimizer_from_config(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.003
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_optical_only_architecture_report_has_no_electronic_nonlinearity():
    report = build_architecture_report(
        model=_ReportModel(),
        config={"readout": {"type": "optical_only"}},
        optimizer_settings={
            "type": "adamw",
            "lr": 0.003,
            "weight_decay": 0.0,
        },
        training_mode="single",
    )
    assert report["electronic_nonlinear_activation_exists"] is False
    assert report["electronic_trainable_parameters_exist"] is False
    assert "only nonlinearity" in report["nonlinearity_statement"]


def test_initial_state_visualizations_are_saved_without_training(tmp_path):
    size = 16
    complex_field = torch.ones(1, size, size, dtype=torch.complex64)
    diagnostics = {
        "amplitudes": torch.ones(4),
        "powers": torch.ones(4),
        "normalized_powers": torch.full((4,), 0.25),
        "expert_energy_ratios": torch.full((4,), 0.25),
        "outside_energy_ratio": 0.0,
        "detector_energies": torch.ones(10),
        "intermediates": {
            "input_amplitude": torch.ones(1, size, size),
            "after_input_to_prompt": complex_field,
            "after_prompt": complex_field,
            "expert_entrance_intensity": torch.ones(1, size, size),
            "after_each_layer": [complex_field for _ in range(5)],
            "after_global_fc": complex_field,
            "detector_intensity": torch.ones(1, size, size),
            "prompt_phase": torch.zeros(size, size),
            "global_fc_phase": torch.zeros(size, size),
        },
    }
    save_initial_state(
        model=_InitialStateModel(),
        diagnostics=diagnostics,
        output_dir=tmp_path,
        val_loss=2.3,
        val_acc=0.1,
    )
    required = [
        "input_amplitude_epoch_0000.png",
        "after_expert_layer_5_epoch_0000.png",
        "detector_plane_epoch_0000.png",
        "prompt_amplitude_bar_epoch_0000.png",
        "expert_phase_layers_epoch_0000.png",
        "initial_diagnostics.json",
    ]
    for filename in required:
        assert (tmp_path / filename).exists()
