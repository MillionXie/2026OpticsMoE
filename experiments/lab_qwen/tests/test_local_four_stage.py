from pathlib import Path
from types import SimpleNamespace

from experiments.lab_qwen.local_four_stage import _configure_local_backbone


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
