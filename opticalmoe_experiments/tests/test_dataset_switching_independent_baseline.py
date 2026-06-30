import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.datasets import DataBundle
from common.utils.config import load_yaml
from dataset_switching.scripts import run_independent_baseline as independent_script
from dataset_switching.scripts.train_dataset_switching import rebuild_dataset_switching_tables


def test_independent_baseline_is_not_upper_bound_in_summary(tmp_path):
    summary = tmp_path / "runs" / "independent" / "summary_for_master"
    summary.mkdir(parents=True)
    rows = [
        {
            "run_id": "independent",
            "task_name": "mnist",
            "is_upper_bound": False,
            "total_independent_params": 123,
        }
    ]
    (summary / "independent_baseline_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    counts = rebuild_dataset_switching_tables(tmp_path / "runs", tmp_path / "results")
    assert counts["independent_baseline"] == 1
    text = (tmp_path / "results" / "master_independent_baseline.csv").read_text(encoding="utf-8")
    assert "False" in text


def test_independent_fast_geometry_parameter_budget_and_task_filter():
    config = load_yaml(ROOT / "dataset_switching" / "configs" / "mnist_fashion_emnist_letters_independent_d2nn.yaml")
    model = independent_script.build_independent_model(config, num_classes=10)
    assert model.canvas_shape == (520, 520)
    assert model.d2nn_phase_grid_size == 360
    assert model.d2nn_local_phase_parameter_count() == 5 * 360 * 360
    assert model.d2nn_global_fc_parameter_count() == 450 * 450
    assert model.optical_parameter_count() == 850500
    assert model.electronic_parameter_count() == 0
    selected = independent_script._task_configs(config, "fashionmnist")
    assert [task["name"] for task in selected] == ["fashionmnist"]


class Args:
    run_name = "independent_smoke"
    task = None
    device = "cpu"
    epochs = 1
    smoke_test = True
    disable_visualization = True


def _fake_bundle(_dataset_cfg, _seed):
    inputs = torch.rand(2, 1, 16, 16)
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    return DataBundle(loader, loader, loader, 10, [str(index) for index in range(10)])


def test_independent_single_task_smoke_writes_parameter_summary(monkeypatch, tmp_path):
    config = {
        "seed": 7,
        "device": "cpu",
        "experiment": {"run_name": "independent_smoke", "independent_group_id": "tiny_group"},
        "model": {
            "type": "independent_d2nn",
            "input_size": 16,
            "canvas_size": 64,
            "d2nn_phase_grid_size": 16,
            "d2nn_num_layers": 1,
            "expected_total_optical_param_count": 512,
            "reference_num_tasks": 3,
            "moe_reference_optical_params": 1536,
        },
        "layout": {
            "canvas_height": 64,
            "canvas_width": 64,
            "input_size": 16,
            "expert_size": 8,
            "expert_pitch": 16,
            "padding": 8,
            "prompt_aperture_size": 48,
        },
        "optics": {
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 8.0e-6,
            "global_fc_phase_mode": "center_window",
            "global_fc_phase_size": 16,
            "distances_m": {
                "input_to_prompt": 0.01,
                "inter_layer": 0.01,
                "layer5_to_fc": 0.01,
                "fc_to_detector": 0.01,
            },
        },
        "detector": {"detector_size": 4, "layout": "grid"},
        "readout": {
            "type": "optical_only",
            "normalize_detector_energy": True,
            "input_norm": "none",
            "norm_affine": False,
            "dropout": 0.0,
        },
        "regularization": {"phase_dropout": {"enabled": False}},
        "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.0},
        "training": {
            "epochs": 1,
            "print_freq": 0,
            "tasks": [
                {
                    "name": "mnist",
                    "dataset": {"name": "mnist", "smoke_train_size": 2, "smoke_test_size": 2},
                    "head": {
                        "detector_size": 4,
                        "detector_layout": "grid",
                        "readout_type": "optical_only",
                        "normalize_detector_energy": True,
                        "logit_scale": 10.0,
                        "input_norm": "none",
                        "norm_affine": False,
                        "hidden_dim": 8,
                        "hidden_layers": 0,
                        "activation": "relu",
                        "dropout": 0.0,
                    },
                }
            ],
        },
        "visualization": {"enabled": False},
        "reporting": {"rebuild_master_tables_after_run": False},
    }
    monkeypatch.setattr(independent_script, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(independent_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(independent_script, "create_dataloaders", _fake_bundle)
    run_dir = independent_script.run_training(config, Args())
    assert (run_dir / "mnist" / "checkpoints" / "best.pt").exists()
    assert (run_dir / "mnist" / "figures" / "training_curves.png").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["planned_total_independent_optical_params"] == 1536
    assert summary["comparison_to_moe_params_ratio"] == 1.0
