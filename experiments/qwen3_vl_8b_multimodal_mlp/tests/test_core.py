from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_8b_multimodal_mlp.features import (
    extract_and_cache,
    multimodal_forward_features,
    pool_answer_hidden_state,
    preprocess_image_text,
)
from experiments.qwen3_vl_8b_multimodal_mlp.benchmark import benchmark_inference
from experiments.qwen3_vl_8b_multimodal_mlp.metrics import classification_metrics
from experiments.qwen3_vl_8b_multimodal_mlp.modeling import MLPHead, parameter_report
from experiments.qwen3_vl_8b_multimodal_mlp.run import (
    _restore_download_cache,
    build_parser,
)
from experiments.qwen3_vl_8b_multimodal_mlp.settings import (
    load_settings,
    normalize_hub_cache_dir,
)
from experiments.qwen3_vl_8b_multimodal_mlp.timing import summarize_timings


class FakeProcessor:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize and add_generation_prompt
        prompt = messages[0]["content"][1]["text"]
        self.prompts.append(prompt)
        return f"<image>{prompt}<assistant>"

    def __call__(self, text, images, return_tensors, padding):
        assert return_tensors == "pt" and padding
        assert len(text) == len(images) == 2
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "pixel_values": torch.zeros(8, 6),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "token_type_ids": torch.zeros(2, 4, dtype=torch.long),
        }


class FakeMultimodalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = type(
            "Config",
            (),
            {
                "vision_config": type(
                    "Vision",
                    (),
                    {"hidden_size": 6, "depth": 2, "out_hidden_size": 8},
                )(),
                "text_config": type(
                    "Text", (), {"hidden_size": 8, "num_hidden_layers": 3}
                )(),
            },
        )()
        self.forward_kwargs: dict | None = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
        return type("Output", (), {"hidden_states": (torch.zeros_like(hidden), hidden)})()


def test_multimodal_processor_forward_and_answer_position() -> None:
    processor = FakeProcessor()
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    prompt = "Classify this image. Answer:"
    inputs = preprocess_image_text(processor, images, prompt)
    assert processor.prompts == [prompt, prompt]
    assert "token_type_ids" not in inputs
    model = FakeMultimodalModel()
    hidden = multimodal_forward_features(model, inputs)
    features, positions = pool_answer_hidden_state(hidden, inputs["attention_mask"])
    assert positions.tolist() == [2, 3]
    assert torch.equal(features[0], hidden[0, 2])
    assert model.forward_kwargs is not None
    assert model.forward_kwargs["output_hidden_states"] is True
    assert model.forward_kwargs["use_cache"] is False


def test_answer_position_supports_left_padding() -> None:
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    _, positions = pool_answer_hidden_state(hidden, mask)
    assert positions.tolist() == [3, 3]


def test_settings_expand_environment_and_reject_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setenv("TEST_HF_HOME", str(tmp_path / "hf"))
    config = config_dir / "test.json"
    config.write_text(
        json.dumps(
            {
                "dataset": "cifar100",
                "data_root": "../data",
                "output_dir": "../runs/test",
                "cache_dir": "${TEST_HF_HOME}",
                "device": "cpu",
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.cache_dir == (tmp_path / "hf").resolve()
    config.write_text(
        json.dumps({"cache_dir": "$DEFINITELY_UNSET_QWEN_CACHE"}), encoding="utf-8"
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


def test_metrics_timing_and_parameter_report() -> None:
    result = classification_metrics(
        labels=[0, 1, 2],
        predictions=[0, 2, 2],
        top5_predictions=[[0, 1], [1, 2], [2, 1]],
        class_names=["a", "b", "c"],
    )
    assert result.top1_accuracy == 2 / 3
    timing = summarize_timings(
        [
            {"samples": 2, "multimodal_forward_sec": 0.2, "end_to_end_sec": 0.4},
            {"samples": 2, "multimodal_forward_sec": 0.4, "end_to_end_sec": 0.6},
        ]
    )
    assert timing["components"]["multimodal_forward_sec"]["mean_per_sample_ms"] == 150.0
    model = FakeMultimodalModel()
    head = MLPHead(8, 4, 3, 0.1)
    report = parameter_report(model, head)
    assert report["mlp_head"]["parameters"] == 51


def test_cli_and_restore_download_cache(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--config", "config.json", "--phase", "download", "--model-id", "Qwen/Qwen3-VL-8B-Instruct"]
    )
    assert args.phase == "download"
    assert args.model_id == "Qwen/Qwen3-VL-8B-Instruct"
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


def test_multimodal_extraction_and_benchmark_outputs(tmp_path: Path) -> None:
    processor = FakeProcessor()
    model = FakeMultimodalModel().eval()
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    labels = torch.tensor([0, 1])
    loader = [(images, labels)]
    prompt = "Classify this image. Answer:"
    features, cached_labels, summary = extract_and_cache(
        model,
        processor,
        loader,
        torch.device("cpu"),
        "train",
        tmp_path,
        {"prompt": prompt},
        "float16",
        False,
        prompt,
    )
    assert list(features.shape) == [2, 8]
    assert torch.equal(cached_labels, labels)
    assert summary["shape"]["answer_hidden_features"] == [2, 8]
    assert summary["shape"]["prompt"] == prompt
    assert (tmp_path / "features" / "train.pt").is_file()

    head = MLPHead(8, 4, 2, 0.0).eval()
    report = benchmark_inference(
        model,
        processor,
        head,
        loader,
        ["zero", "one"],
        torch.device("cpu"),
        tmp_path,
        warmup_batches=1,
        benchmark_batches=None,
        progress=False,
        classification_prompt=prompt,
    )
    assert report["feature_shapes"]["last_hidden_state"] == [2, 4, 8]
    assert "multimodal_forward_sec" in report["timing"]["components"]
    for name in (
        "inference.json",
        "inference_batches.csv",
        "predictions.csv",
        "confusion_matrix.csv",
    ):
        assert (tmp_path / "metrics" / name).is_file()
