from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from experiments.qwen_vl_cifar10.config import parse_args_with_config
from experiments.qwen_vl_cifar10.inference_benchmark import (
    _infer_images,
    _prepare_inputs,
    _validate_args,
    _validate_gpu_capacity,
    build_parser,
)
from experiments.qwen_vl_cifar10.inference_profiling import (
    BatchTiming,
    audit_model_features,
    parameter_statistics,
    summarize_batch_timings,
    timed_processor_components,
)
from experiments.qwen_vl_cifar10.visualize_inference import write_inference_figure


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FakeComponent:
    def __init__(self, result: dict[str, torch.Tensor]) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls += 1
        return self.result


class FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = FakeComponent({"pixel_values": torch.ones(1, 4)})
        self.tokenizer = FakeComponent(
            {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        )

    def apply_chat_template(
        self, conversations: list[object], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        batch_size = len(conversations)
        self.image_processor(images=conversations)
        self.tokenizer(["prompt"] * batch_size)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
            "pixel_values": torch.ones(batch_size, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size),
        }

    def batch_decode(self, tokens: torch.Tensor, **kwargs: object) -> list[str]:
        return ["cat"] * len(tokens)


class FakeGenerateModel(nn.Module):
    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        suffix = torch.full(
            (len(input_ids), 1), 2, dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat((input_ids, suffix), dim=1)


class FakeMLP(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate)
        self.down_proj = nn.Linear(intermediate, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(value)))


class FakeTextLayer(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = FakeMLP(width, intermediate)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mlp(value)


class FakeVisionBlock(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = SimpleNamespace(linear_fc1=nn.Linear(width, intermediate))
        self.down = nn.Linear(intermediate, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.down(torch.relu(self.mlp.linear_fc1(value)))


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(4, 6)
        self.blocks = nn.ModuleList([FakeVisionBlock(6, 10) for _ in range(4)])
        self.merger = nn.Linear(6, 8)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        value = self.patch_embed(pixels)
        for block in self.blocks:
            value = block(value)
        return self.merger(value)


class FakeLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeTextLayer(8, 12) for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = FakeVisual()
        self.language_model = FakeLanguage()
        self.embedding = nn.Embedding(16, 8)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        hidden = self.language_model(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class FakeAuditModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeCore()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=8,
                intermediate_size=12,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                vocab_size=16,
            ),
            vision_config=SimpleNamespace(
                hidden_size=6,
                intermediate_size=10,
                depth=4,
                num_heads=2,
                patch_size=2,
                spatial_merge_size=1,
                temporal_patch_size=1,
                out_hidden_size=8,
                deepstack_visual_indexes=[1, 2],
            ),
        )

    def get_base_model(self) -> "FakeAuditModel":
        return self

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        values = self.model.visual(pixel_values)
        per_image = tuple(value.unsqueeze(0) for value in values)
        return per_image, [values]


def _batch_timing(index: int, scale: float = 1.0) -> BatchTiming:
    return BatchTiming(
        batch_index=index,
        image_count=1,
        input_tokens=3,
        generated_tokens=1,
        dataset_fetch_sec=0.001 * scale,
        prompt_build_sec=0.001 * scale,
        image_preprocess_sec=0.002 * scale,
        tokenizer_sec=0.001 * scale,
        processor_framework_sec=0.001 * scale,
        processor_total_sec=0.004 * scale,
        host_to_device_sec=0.001 * scale,
        model_generate_sec=0.01 * scale,
        decode_postprocess_sec=0.001 * scale,
        complete_inference_sec=0.018 * scale,
        end_to_end_sec=0.019 * scale,
    )


def test_h20_configs_parse() -> None:
    for name, limit in (
        ("inference_32b_h20.json", None),
        ("inference_32b_h20_smoke.json", 32),
    ):
        args = parse_args_with_config(
            build_parser(), ["--config", str(CONFIG_DIR / name)]
        )
        _validate_args(args)
        assert args.model_id == "Qwen/Qwen3-VL-32B-Instruct"
        assert args.batch_size == 1
        assert args.test_limit == limit
        assert args.require_all_cuda
        assert args.min_gpu_memory_gib == 80.0


def test_h20_preflight_rejects_small_gpu_slice() -> None:
    with (
        patch("torch.cuda.device_count", return_value=1),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(total_memory=48 * 1024**3),
        ),
        pytest.raises(RuntimeError, match="full 96 GB device"),
    ):
        _validate_gpu_capacity(torch.device("cuda:0"), None, 80.0)

    with (
        patch("torch.cuda.device_count", return_value=1),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(total_memory=96 * 1024**3),
        ),
    ):
        _validate_gpu_capacity(torch.device("cuda:0"), None, 80.0)


def test_processor_component_timer_and_inference_split() -> None:
    processor = FakeProcessor()
    with timed_processor_components(processor) as (image_timer, tokenizer_timer):
        processor.image_processor(images=["image"])
        processor.tokenizer(["prompt"])
    assert image_timer.calls == 1
    assert tokenizer_timer.calls == 1
    assert processor.image_processor.calls == 1
    assert processor.tokenizer.calls == 1

    outputs, timing = _infer_images(
        FakeGenerateModel(),
        processor,
        [object()],
        ["cat", "dog"],
        torch.device("cpu"),
        max_new_tokens=1,
        batch_index=0,
        dataset_fetch_sec=0.001,
        end_to_end_start=__import__("time").perf_counter(),
    )
    assert outputs == ["cat"]
    assert timing.input_tokens == 3
    assert timing.generated_tokens == 1
    assert timing.image_preprocess_sec >= 0
    assert timing.tokenizer_sec >= 0
    assert timing.model_generate_sec > 0
    assert timing.end_to_end_sec >= timing.complete_inference_sec


def test_timing_statistics_include_percentiles() -> None:
    summary = summarize_batch_timings([_batch_timing(0, 1.0), _batch_timing(1, 2.0)])
    model = summary["model_generate_sec"]
    assert model["images"] == 2
    assert model["total_sec"] == 0.03
    assert model["p95_ms_per_image"] > model["median_ms_per_image"]
    assert model["images_per_second"] > 0


def test_feature_audit_records_vision_text_and_pooled_shapes() -> None:
    model = FakeAuditModel()
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.ones(1, 4),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    audit = audit_model_features(model, inputs, torch.device("cpu"))
    assert audit["feature_dimension"] == 8
    assert audit["pooled_feature"]["shape"] == [1, 8]
    assert audit["architecture"]["vision"]["hidden_size"] == 6
    assert audit["architecture"]["text"]["hidden_size"] == 8
    assert "vision.patch_embed" in audit["intermediate_modules"]
    assert "text.layers.0.mlp.gate_proj" in audit["intermediate_modules"]
    stats = parameter_statistics(model)
    assert stats["total_parameters"] > 0
    assert stats["parameter_bytes"] > 0


def test_inference_figure_is_written(tmp_path: Path) -> None:
    timings = [_batch_timing(0), _batch_timing(1, 1.2)]
    metrics = {
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "accuracy": 0.8,
        "macro_f1": 0.79,
        "per_class_accuracy": {"cat": 0.9, "dog": 0.7},
        "timing_summary": summarize_batch_timings(timings),
    }
    audit = {
        "intermediate_modules": {
            "vision.blocks.0": {
                "observations": [{"shape": [256, 1152], "numel": 294912}]
            }
        },
        "pooled_feature": {"shape": [1, 5120]},
    }
    path = tmp_path / "inference_summary.png"
    write_inference_figure(metrics, timings, audit, path)
    assert path.is_file()
    assert path.with_suffix(".pdf").is_file()
