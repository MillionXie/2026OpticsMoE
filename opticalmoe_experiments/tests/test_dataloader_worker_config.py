import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.loader_utils import apply_smoke_loader_overrides, dataloader_kwargs


def test_dataloader_kwargs_single_process_omits_worker_only_args():
    cfg = {
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    kwargs = dataloader_kwargs(cfg, shuffle=True, seed=7)
    assert kwargs["batch_size"] == 8
    assert kwargs["shuffle"] is True
    assert kwargs["num_workers"] == 0
    assert kwargs["pin_memory"] is False
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs
    assert "generator" in kwargs


def test_dataloader_kwargs_multi_worker_keeps_worker_args():
    cfg = {
        "batch_size": 16,
        "num_workers": 16,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    kwargs = dataloader_kwargs(cfg, shuffle=False)
    assert kwargs["batch_size"] == 16
    assert kwargs["num_workers"] == 16
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_smoke_loader_override_disables_worker_args():
    cfg = {"num_workers": 16, "persistent_workers": True, "prefetch_factor": 4}
    apply_smoke_loader_overrides(cfg)
    assert cfg["num_workers"] == 0
    assert cfg["persistent_workers"] is False
    assert cfg["prefetch_factor"] is None
    kwargs = dataloader_kwargs(cfg)
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_configs_expose_worker_fields():
    paths = [
        ROOT / "single_task" / "configs" / "mnist_learnable_moe_E9_complex.yaml",
        ROOT / "dataset_switching" / "configs" / "mnist_fashion_emnist_letters_learnable_moe_E9_complex.yaml",
        ROOT / "same_input_multitask" / "configs" / "dsprites_shape_scale_learnable_moe_E9_complex.yaml",
    ]
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset_cfgs = []
        if "dataset" in cfg:
            dataset_cfgs.append(cfg["dataset"])
        for task in cfg.get("training", {}).get("multitask", {}).get("tasks", []):
            dataset_cfgs.append(task["dataset"])
        assert dataset_cfgs, path
        for dataset_cfg in dataset_cfgs:
            assert dataset_cfg["num_workers"] == 16
            assert dataset_cfg["pin_memory"] == "auto"
            assert dataset_cfg["persistent_workers"] is True
            assert dataset_cfg["prefetch_factor"] == 4
