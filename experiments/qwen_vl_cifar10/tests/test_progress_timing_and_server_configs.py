from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import call, patch

import torch

from experiments.qwen_vl_cifar10.benchmark import benchmark_callable
from experiments.qwen_vl_cifar10.config import parse_args_with_config
from experiments.qwen_vl_cifar10.main import (
    _resolve_batch_sizes,
    _validate_args,
    build_parser,
)
from experiments.qwen_vl_cifar10.models import MLPHead, SUPPORTED_MODEL_IDS
from experiments.qwen_vl_cifar10.progress import progress_iter
from experiments.qwen_vl_cifar10.train import train_mlp_head
from experiments.qwen_vl_cifar10.utils import reset_cuda_peak_memory
from experiments.qwen_vl_cifar10.visualize_results import (
    write_comparison_figure,
    write_run_figure,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_server_configs_parse_and_use_split_batch_sizes() -> None:
    expected = {
        "mlp_4b_server_1x4090.json": ("Qwen/Qwen3-VL-4B-Instruct", "none"),
        "mlp_8b_server_1x4090.json": ("Qwen/Qwen3-VL-8B-Instruct", "none"),
        "mlp_30b_a3b_server_4x4090.json": (
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "auto",
        ),
        "mlp_32b_server_4x4090.json": ("Qwen/Qwen3-VL-32B-Instruct", "auto"),
    }
    for filename, (model_id, device_map) in expected.items():
        args = parse_args_with_config(
            build_parser(), ["--config", str(CONFIG_DIR / filename)]
        )
        _resolve_batch_sizes(args)
        _validate_args(args)
        assert args.model_id == model_id
        assert args.model_id in SUPPORTED_MODEL_IDS
        assert args.device_map == device_map
        assert args.feature_batch_size <= 2
        assert args.head_batch_size == 512


def test_mlp_training_records_epoch_train_and_evaluation_time() -> None:
    generator = torch.Generator().manual_seed(7)
    train_features = torch.randn(24, 8, generator=generator)
    train_labels = torch.arange(24) % 3
    test_features = torch.randn(12, 8, generator=generator)
    test_labels = torch.arange(12) % 3
    class_names = ["zero", "one", "two"]
    head = MLPHead(feature_dim=8, hidden_dim=4)
    head.network[-1] = torch.nn.Linear(4, len(class_names))

    result = train_mlp_head(
        head,
        train_features,
        train_labels,
        test_features,
        test_labels,
        class_names,
        torch.device("cpu"),
        batch_size=6,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=7,
        show_progress=False,
    )

    assert result.elapsed_sec > 0
    assert result.evaluation_elapsed_sec > 0
    assert len(result.history) == 2
    assert all(row["train_time_sec"] > 0 for row in result.history)
    assert all(row["evaluation_time_sec"] > 0 for row in result.history)


def test_progress_can_be_disabled() -> None:
    values = [1, 2, 3]
    assert list(progress_iter(values, description="test", enabled=False)) == values


def test_benchmark_records_warmup_and_measurement_durations() -> None:
    result = benchmark_callable(
        lambda: 2,
        torch.device("cpu"),
        warmup_batches=1,
        benchmark_batches=3,
    )
    assert result.first_batch_latency_sec >= 0
    assert result.warmup_time_sec >= 0
    assert result.measurement_time_sec > 0
    assert result.measured_images == 6
    assert result.total_time_sec >= result.measurement_time_sec


def test_peak_memory_reset_initializes_visible_cuda_contexts_first() -> None:
    with (
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.synchronize") as synchronize,
        patch("torch.cuda.reset_peak_memory_stats") as reset,
    ):
        reset_cuda_peak_memory(torch.device("cuda"))

    assert synchronize.call_args_list == [call(0), call(1)]
    assert reset.call_args_list == [call(0), call(1)]


def test_run_and_comparison_figures_are_written(tmp_path: Path) -> None:
    metrics = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "mode": "mlp",
        "accuracy": 0.75,
        "macro_f1": 0.73,
        "per_class_accuracy": {"cat": 0.8, "dog": 0.7},
        "total_wall_time_sec": 120.0,
        "model_load_time_sec": 10.0,
        "feature_extraction_total_sec": 80.0,
        "head_train_total_sec": 20.0,
        "evaluation_total_sec": 5.0,
        "end_to_end_images_per_second": 12.0,
        "cuda_peak_memory_mb": 4096.0,
        "feature_batch_size": 2,
        "head_batch_size": 512,
        "seed": 42,
        "training_history": [
            {
                "epoch": 1,
                "train_loss": 1.2,
                "eval_loss": 1.0,
                "accuracy": 0.7,
                "macro_f1": 0.68,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "eval_loss": 0.7,
                "accuracy": 0.75,
                "macro_f1": 0.73,
            },
        ],
        "timing": {
            "stages": {
                "dataset_load_sec": 5.0,
                "model_load_sec": 10.0,
                "feature_extraction_total_sec": 80.0,
                "head_or_adapter_train_sec": 20.0,
                "evaluation_total_sec": 5.0,
                "generation_total_sec": 0.0,
            }
        },
    }
    run_path = tmp_path / "run_summary.png"
    write_run_figure(metrics, run_path)
    write_comparison_figure([metrics], tmp_path / "summary")

    assert run_path.is_file()
    assert run_path.with_suffix(".pdf").is_file()
    assert (tmp_path / "summary" / "qwen_vl_model_comparison.png").is_file()
    assert (tmp_path / "summary" / "qwen_vl_model_comparison.pdf").is_file()
    csv_path = tmp_path / "summary" / "results_summary.csv"
    assert csv_path.is_file()
    assert "Qwen/Qwen3-VL-2B-Instruct" in csv_path.read_text(encoding="utf-8")


def test_existing_2b_config_uses_fast_head_batch() -> None:
    with (CONFIG_DIR / "mlp_2b.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["feature_batch_size"] == 2
    assert config["head_batch_size"] == 512
