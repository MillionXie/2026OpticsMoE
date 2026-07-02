from __future__ import annotations

from pathlib import Path

import torch

from experiments.qwen_vl_8B.io_utils import write_json
from experiments.qwen_vl_8B.modeling import MLPHead
from experiments.qwen_vl_8B.training import train_head
from experiments.qwen_vl_8B.visualize import generate_figures


def test_train_head_and_generate_figures(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(80, 6, generator=generator)
    labels = (features[:, 0] > 0).long()
    test_features = torch.randn(20, 6, generator=generator)
    test_labels = (test_features[:, 0] > 0).long()
    head = MLPHead(6, 8, 2, 0.0)
    head, report = train_head(
        head,
        features,
        labels,
        test_features,
        test_labels,
        ["negative", "positive"],
        torch.device("cpu"),
        tmp_path,
        batch_size=16,
        epochs=2,
        validation_fraction=0.2,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=2,
        progress=False,
    )
    assert report["best_epoch"] in {1, 2}
    inference = {
        "metrics": report["test_metrics"],
        "timing": {
            "components": {
                name: {"mean_per_sample_ms": value}
                for name, value in {
                    "data_loading_sec": 0.1,
                    "image_preprocess_sec": 0.2,
                    "host_to_device_sec": 0.1,
                    "vision_forward_sec": 2.0,
                    "pooling_sec": 0.1,
                    "mlp_forward_sec": 0.1,
                    "postprocess_sec": 0.1,
                }.items()
            }
        },
    }
    write_json(tmp_path / "metrics" / "inference.json", inference)
    outputs = generate_figures(tmp_path)
    assert outputs
    assert all(path.is_file() for path in outputs)

