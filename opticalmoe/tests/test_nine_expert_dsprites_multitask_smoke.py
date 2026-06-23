import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from opticalmoe.optics.nine_expert_as_multitask_moe import (
    NineExpertASGlobalRouterMultitaskMoEClassifier,
)


def _write_fake_dsprites(path: Path, n: int = 48):
    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 2, size=(n, 64, 64), dtype=np.uint8)
    latents_classes = np.zeros((n, 6), dtype=np.int64)
    latents_classes[:, 1] = np.arange(n) % 3
    latents_classes[:, 2] = np.arange(n) % 6
    latents_values = latents_classes.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, imgs=imgs, latents_classes=latents_classes, latents_values=latents_values)
    return path


def _load_dsprites_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_nine_expert_dsprites_multitask_moe.py"
    spec = importlib.util.spec_from_file_location("train_nine_expert_dsprites_multitask_moe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nine_expert_dsprites_shape_scale_forward():
    model = NineExpertASGlobalRouterMultitaskMoEClassifier(
        task_names=["shape", "scale"],
        task_num_classes={"shape": 3, "scale": 6},
        task_head_configs={
            "shape": {"readout_type": "mlp", "hidden_dim": 8},
            "scale": {"readout_type": "mlp", "hidden_dim": 8},
        },
        num_layers=1,
        expert_phase_init="identity",
        global_fc_phase_init="identity",
    )
    images = torch.rand(1, 1, 134, 134)
    logits_shape, intermediates = model(images, task_name="shape", return_intermediates=True)
    logits_scale = model(images, task_name="scale")

    assert logits_shape.shape == (1, 3)
    assert logits_scale.shape == (1, 6)
    assert "prompt_router_amplitude" in intermediates
    assert "expert_entrance_after_aperture" in intermediates


class TinyTaskModel(nn.Module):
    def forward(self, images, task_name=None, **_kwargs):
        batch = images.shape[0]
        if task_name == "shape":
            return torch.zeros(batch, 3, device=images.device)
        if task_name == "scale":
            return torch.zeros(batch, 6, device=images.device)
        raise KeyError(task_name)


def test_same_input_task_switching_evaluation_outputs(tmp_path):
    script = _load_dsprites_script()
    npz_path = _write_fake_dsprites(tmp_path / "fake_dsprites.npz")
    config = {
        "seed": 7,
        "visualization": {"enabled": False, "num_samples": 1},
        "training": {
            "multitask": {
                "tasks": [
                    {
                        "name": "shape",
                        "dataset": {
                            "name": "dsprites",
                            "task": "shape",
                            "root": str(tmp_path),
                            "npz_path": str(npz_path),
                            "input_size": 134,
                            "download": False,
                        },
                    },
                    {
                        "name": "scale",
                        "dataset": {
                            "name": "dsprites",
                            "task": "scale",
                            "root": str(tmp_path),
                            "npz_path": str(npz_path),
                            "input_size": 134,
                            "download": False,
                        },
                    },
                ]
            }
        },
    }
    result = script.same_input_task_switching_evaluation(
        TinyTaskModel(),
        config,
        tmp_path,
        torch.device("cpu"),
        max_samples=4,
    )

    assert 0.0 <= result["shape_accuracy"] <= 1.0
    assert 0.0 <= result["scale_accuracy"] <= 1.0
    assert (tmp_path / "same_input_task_switching.csv").exists()
    assert (tmp_path / "same_input_task_switching.json").exists()
