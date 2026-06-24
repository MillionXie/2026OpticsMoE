import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.electronic_models import LeNet5Classifier


def test_lenet_forward_supports_multiple_input_sizes():
    for input_size in (134, 256):
        model = LeNet5Classifier(num_classes=10, input_size=input_size)
        logits = model(torch.rand(2, 1, input_size, input_size))
        assert logits.shape == (2, 10)
