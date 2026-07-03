from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest
from torch import nn

from experiments.qwen3_vl_8b_vision_mlp.features import image_token_features, pool_tokens
from experiments.qwen3_vl_8b_vision_mlp.metrics import classification_metrics
from experiments.qwen3_vl_8b_vision_mlp.modeling import MLPHead, parameter_report
from experiments.qwen3_vl_8b_vision_mlp.settings import (
    load_settings,
    normalize_hub_cache_dir,
    resolve_model_id,
)
from experiments.qwen3_vl_8b_vision_mlp.timing import summarize_timings
from experiments.qwen3_vl_8b_vision_mlp.run import _restore_download_cache, build_parser


class FakeVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = type(
            "Config",
            (),
            {
                "vision_config": type(
                    "Vision", (), {"spatial_merge_size": 2, "hidden_size": 6, "depth": 2,
                                   "out_hidden_size": 8}
                )(),
                "text_config": type("Text", (), {"hidden_size": 8, "num_hidden_layers": 3})(),
            },
        )()

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        del pixel_values
        total = sum(int(value) // 4 for value in image_grid_thw.prod(dim=-1))
        return torch.arange(total * 8, dtype=torch.float32).reshape(total, 8), []


class FakeTransformers5VisionModel(FakeVisionModel):
    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        del pixel_values
        batch_size = image_grid_thw.shape[0]
        merged = tuple(torch.ones(4, 8) * index for index in range(batch_size))
        return type(
            "VisionOutput",
            (),
            {
                "last_hidden_state": torch.zeros(batch_size * 16, 6),
                "pooler_output": merged,
            },
        )()


def test_settings_resolve_paths(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "test.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "dataset": "cifar100",
                "data_root": "../data",
                "output_dir": "../runs/test",
                "device": "cpu",
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.data_root == (tmp_path / "data").resolve()
    assert settings.output_dir == (tmp_path / "runs" / "test").resolve()
    assert settings.model_id == "Qwen/Qwen3-VL-8B-Instruct"


def test_visual_token_split_and_pool() -> None:
    model = FakeVisionModel()
    inputs = {
        "pixel_values": torch.zeros(32, 3),
        "image_grid_thw": torch.tensor([[1, 8, 2], [1, 4, 4]]),
    }
    tokens = image_token_features(model, inputs)
    assert [list(value.shape) for value in tokens] == [[4, 8], [4, 8]]
    assert list(pool_tokens(tokens).shape) == [2, 8]


def test_transformers5_prefers_merged_pooler_output() -> None:
    model = FakeTransformers5VisionModel()
    inputs = {
        "pixel_values": torch.zeros(32, 3),
        "image_grid_thw": torch.tensor([[1, 8, 2], [1, 4, 4]]),
    }
    tokens = image_token_features(model, inputs)
    assert [list(value.shape) for value in tokens] == [[4, 8], [4, 8]]
    assert torch.equal(tokens[1], torch.ones(4, 8))


def test_metrics_and_timing() -> None:
    result = classification_metrics(
        labels=[0, 1, 2],
        predictions=[0, 2, 2],
        top5_predictions=[[0, 1], [1, 2], [2, 1]],
        class_names=["a", "b", "c"],
    )
    assert result.top1_accuracy == 2 / 3
    assert result.top5_accuracy == 1.0
    timing = summarize_timings(
        [
            {"samples": 2, "vision_forward_sec": 0.2, "end_to_end_sec": 0.4},
            {"samples": 2, "vision_forward_sec": 0.4, "end_to_end_sec": 0.6},
        ]
    )
    assert timing["samples"] == 4
    assert timing["throughput_images_per_sec"] == 4.0
    assert timing["components"]["vision_forward_sec"]["mean_per_sample_ms"] == 150.0


def test_parameter_report_includes_head() -> None:
    model = FakeVisionModel()
    head = MLPHead(8, 4, 3, 0.1)
    report = parameter_report(model, head)
    assert report["mlp_head"]["parameters"] == 51
    assert report["architecture"]["vision_hidden_size"] == 6


def test_download_phase_cli() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.json",
            "--phase",
            "download",
            "--model-id",
            "Qwen/Qwen3-VL-8B-Instruct",
        ]
    )
    assert args.phase == "download"
    assert args.download_workers == 2
    assert args.model_id == "Qwen/Qwen3-VL-8B-Instruct"


def test_environment_and_relative_local_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    model_dir = config_dir / "local-model"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("VISION_CACHE", str(tmp_path / "cache"))
    config = config_dir / "test.json"
    config.write_text(
        json.dumps(
            {
                "model_id": "local-model",
                "cache_dir": "$VISION_CACHE",
                "device": "cpu",
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.model_id == str(model_dir.resolve())
    assert settings.cache_dir == (tmp_path / "cache").resolve()
    assert resolve_model_id("local-model", config_dir) == str(model_dir.resolve())
    config.write_text(
        json.dumps({"cache_dir": "${UNSET_VISION_CACHE_FOR_TEST}"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unset environment variable"):
        load_settings(config)


def test_hf_home_root_resolves_to_nested_hub(tmp_path: Path) -> None:
    hf_home = tmp_path / "huggingface"
    repo = hf_home / "hub" / "models--Qwen--Qwen3-VL-8B-Instruct"
    repo.mkdir(parents=True)
    assert normalize_hub_cache_dir(
        hf_home, "Qwen/Qwen3-VL-8B-Instruct"
    ) == (hf_home / "hub").resolve()


def test_restore_download_cache_from_legacy_record(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = cache_dir / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "download.json").write_text(
        json.dumps({"snapshot": str(snapshot)}), encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": "cifar100",
                "data_root": str(tmp_path / "data"),
                "output_dir": str(output_dir),
                "local_files_only": True,
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    _restore_download_cache(settings, "all")
    assert settings.cache_dir == cache_dir.resolve()
