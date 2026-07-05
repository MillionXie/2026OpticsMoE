import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.electronic_baselines import SupervisedLeNetClassifier


def test_supervised_lenet_forward_without_teacher_or_projector():
    model = SupervisedLeNetClassifier(
        num_classes=10,
        lenet_config={
            "channels": [4, 8, 16], "output_feature_dim": 900,
            "conv_dropout2d": 0.1, "feature_dropout": 0.2,
        },
        feature_preprocess_config={"norm": "layernorm", "activation": "gelu"},
        classifier_config={
            "input": "lenet_feature", "input_dim": 900,
            "hidden_layers": 1, "hidden_dim": 16, "dropout": 0.2,
        },
    )
    logits, raw, processed = model(torch.rand(2, 1, 48, 48))
    assert logits.shape == (2, 10)
    assert raw.shape == processed.shape == (2, 900)
    assert not hasattr(model, "projector")
    assert model.total_parameter_count() == sum(parameter.numel() for parameter in model.parameters())
