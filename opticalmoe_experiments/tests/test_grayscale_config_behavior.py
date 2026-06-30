import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.transforms import build_image_transform


def test_grayscale_flag_false_still_outputs_single_channel_for_mnist_like_input():
    image = Image.new("L", (28, 28), color=128)
    out_true = build_image_transform(120, grayscale=True)(image)
    out_false = build_image_transform(120, grayscale=False)(image)
    assert tuple(out_true.shape) == (1, 120, 120)
    assert tuple(out_false.shape) == (1, 120, 120)


def test_cifar_rgb_with_grayscale_false_currently_outputs_single_channel():
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    out = build_image_transform(120, grayscale=False)(image)
    assert tuple(out.shape) == (1, 120, 120)
