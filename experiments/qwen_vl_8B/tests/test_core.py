from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from experiments.qwen_vl_8B.features import image_token_features, pool_tokens
from experiments.qwen_vl_8B.metrics import classification_metrics
from experiments.qwen_vl_8B.modeling import MLPHead, parameter_report
from experiments.qwen_vl_8B.settings import load_settings
from experiments.qwen_vl_8B.timing import summarize_timings
from experiments.qwen_vl_8B.run import build_parser


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
    args = build_parser().parse_args(["--config", "config.json", "--phase", "download"])
    assert args.phase == "download"
    assert args.download_workers == 2
