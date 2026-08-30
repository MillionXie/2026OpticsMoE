from pathlib import Path
from types import SimpleNamespace

from experiments.lab_qwen.local_four_stage import (
    PROFILES,
    _configure_local_backbone,
    _paths,
)


def test_local_four_stage_prefers_complete_offline_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "models/Qwen3-VL-Embedding-2B"
    model.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    ):
        (model / name).write_bytes(b"present")
    settings = SimpleNamespace(
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        cache_dir=tmp_path / "cache",
        local_files_only=False,
    )

    report = _configure_local_backbone(settings, tmp_path, None)

    assert report["mode"] == "bundled_local_snapshot"
    assert settings.model_id == str(model.resolve())
    assert settings.cache_dir is None
    assert settings.local_files_only is True


def test_tradeoff_profiles_never_share_checkpoint_or_session_paths(
    tmp_path: Path,
) -> None:
    outputs = {
        profile: _paths(tmp_path, "vision_expert", profile)
        for profile in PROFILES
    }
    assert len({str(paths[1]) for paths in outputs.values()}) == len(PROFILES)
    assert len({str(paths[2]) for paths in outputs.values()}) == len(PROFILES)
    assert outputs["accuracy_first"][2].name == "accuracy_first_ema.pt"
    assert outputs["balanced"][2].name == "balanced_ema.pt"
