from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_all_foundation_training_configs_use_camera_grid30():
    configs = sorted((ROOT / "foundation_distillation" / "configs").glob("*.yaml"))
    assert configs
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        feature = config["feature_detector"]
        assert feature["source_region"] == "camera_active_window", path
        assert feature["grid_size"] == 30, path
        assert feature["feature_dim"] == 900, path
        if "teacher" in config:
            assert config["classifier"]["input_dim"] == "auto_teacher_dim", path
            assert config["projector"]["input_dim"] == "auto_feature_dim", path
            assert config["projector"]["output_dim"] == "auto_teacher_dim", path
            assert config["loss"]["leak_loss_weight"] == 0.0, path
