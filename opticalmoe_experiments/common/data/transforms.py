from torchvision import transforms
from torchvision.transforms import functional as TF
import torch


class PILToFloatTensorNoNumpy:
    """Convert a PIL image to [1,H,W] float tensor without numpy."""

    def __call__(self, image):
        image = image.convert("L")
        width, height = image.size
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(height, width).unsqueeze(0)
        return tensor.to(dtype=torch.float32).div_(255.0)


class FixEMNISTOrientation:
    """Rotate/flip EMNIST glyphs into the common upright convention."""

    def __call__(self, image):
        return TF.hflip(TF.rotate(image, -90))


def build_image_transform(
    input_size: int,
    grayscale: bool = True,
    fix_emnist_orientation: bool = False,
):
    steps = []
    if fix_emnist_orientation:
        steps.append(FixEMNISTOrientation())
    if grayscale:
        steps.append(transforms.Grayscale(num_output_channels=1))
    steps.extend(
        [
            transforms.Resize((int(input_size), int(input_size))),
            PILToFloatTensorNoNumpy(),
        ]
    )
    return transforms.Compose(steps)

