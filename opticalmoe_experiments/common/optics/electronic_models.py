import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5Classifier(nn.Module):
    """Simple electronic LeNet-style baseline for grayscale inputs."""

    def __init__(self, num_classes: int, input_channels: int = 1, input_size: int = 120) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, int(num_classes)),
        )

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        images = F.interpolate(images.float(), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        features = self.features(images)
        features = self.adaptive_pool(features)
        logits = self.classifier(features.flatten(1))
        if return_intermediates:
            return logits, {"input_amplitude": images[:, 0], "features": features, "logits": logits}
        return logits

    def optical_parameter_count(self) -> int:
        return 0

    def prompt_parameter_count(self) -> int:
        return 0

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_phase_dropout_active(self, active: bool) -> None:
        return None
