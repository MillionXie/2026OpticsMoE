import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.electronic_baselines import FeatureDistilledLeNetClassifier


def test_lenet_dropout_is_configurable_and_reported():
    model = FeatureDistilledLeNetClassifier(
        num_classes=10,
        teacher_feature_dim=32,
        lenet_config={
            "input_channels": 1,
            "channels": [4, 8, 16],
            "adaptive_pool_size": 5,
            "output_feature_dim": 900,
            "conv_dropout2d": 0.1,
            "feature_dropout": 0.2,
        },
        projector_config={"type": "linear", "input_dim": 900, "output_dim": "auto_teacher_dim"},
        classifier_config={"input": "semantic_feature", "input_dim": "auto_teacher_dim"},
    )
    dropouts2d = [module for module in model.lenet_backbone if isinstance(module, nn.Dropout2d)]
    assert len(dropouts2d) == 2
    assert all(module.p == 0.1 for module in dropouts2d)
    assert isinstance(model.feature_dropout, nn.Dropout)
    assert model.feature_dropout.p == 0.2
    assert model.lenet_config["conv_dropout2d"] == 0.1
    assert model.lenet_config["feature_dropout"] == 0.2
    model.eval()
    with torch.inference_mode():
        logits, raw, processed, semantic, normalized = model(torch.rand(2, 1, 32, 32))
    assert logits.shape == (2, 10)
    assert raw.shape == processed.shape == (2, 900)
    assert semantic.shape == normalized.shape == (2, 32)


def test_lenet_dropout_defaults_preserve_old_configs():
    model = FeatureDistilledLeNetClassifier(
        num_classes=3,
        teacher_feature_dim=8,
        lenet_config={"channels": [4, 8, 16], "output_feature_dim": 900},
        projector_config={"type": "linear", "input_dim": 900, "output_dim": "auto_teacher_dim"},
        classifier_config={"input": "semantic_feature", "input_dim": "auto_teacher_dim"},
    )
    assert model.lenet_config["conv_dropout2d"] == 0.0
    assert model.lenet_config["feature_dropout"] == 0.0
