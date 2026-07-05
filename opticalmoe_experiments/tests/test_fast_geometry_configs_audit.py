import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dataset_blocks(config):
    if isinstance(config.get("dataset"), dict):
        yield config["dataset"]
    for task in config.get("training", {}).get("multitask", {}).get("tasks", []):
        if isinstance(task, dict) and isinstance(task.get("dataset"), dict):
            yield task["dataset"]


def test_all_new_training_configs_use_fast120_520():
    families = ("single_task", "dataset_switching", "same_input_multitask", "foundation_distillation")
    checked = 0
    for family in families:
        for path in (ROOT / family / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if config.get("student", {}).get("model_type") in {"feature_distilled_lenet", "supervised_lenet"}:
                assert config["dataset"]["input_size"] == 120
                assert config["dataset"]["grayscale"] is True
                continue
            if config.get("model", {}).get("type") == "lenet5":
                assert config["dataset"]["input_size"] == 120
                continue
            layout = config.get("layout", {})
            assert layout == {
                "geometry_profile": "fast120_520",
                "canvas_height": 520,
                "canvas_width": 520,
                "input_size": 120,
                "expert_size": 120,
                "expert_pitch": 150,
                "padding": 35,
                "prompt_aperture_size": 450,
            }, path
            optics = config.get("optics", {})
            assert optics.get("global_fc_phase_mode") == "center_window", path
            assert optics.get("global_fc_phase_size") == 450, path
            assert optics.get("global_fc_padding_mode") == "transparent", path
            for dataset in _dataset_blocks(config):
                assert dataset.get("input_size") == 120, path
            checked += 1
    # Includes the two DINOv2-base foundation-distillation configurations.
    assert checked == 45


def test_transfer_source_config_remains_legacy():
    path = ROOT / "transfer_adaptation" / "pretrained_backbones" / "dataset_switching_moe_mnist_fashion_emnist_letters" / "source_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["layout"]["canvas_height"] == 1000
    assert config["layout"]["input_size"] == 134
    assert config["optics"]["global_fc_phase_size"] == 600
