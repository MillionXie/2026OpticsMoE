"""Shared electronic detector readout used by the MoE and D2NN baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class ElectronicDetectorReadout(nn.Module):
    """A 10-to-10 linear classifier on log relative detector energies.

    Identity initialization preserves the optical detector argmax at the start
    of training while still allowing a small electronic stage to calibrate
    detector gains and learn cross-detector combinations.
    """

    def __init__(self, num_classes: int, eps: float = 1.0e-8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.eps <= 0:
            raise ValueError("readout eps must be positive")
        self.classifier = nn.Linear(self.num_classes, self.num_classes, bias=True)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(self.num_classes))
            self.classifier.bias.zero_()

    def features(self, detector_energies: torch.Tensor) -> torch.Tensor:
        probabilities = (detector_energies + self.eps) / (
            detector_energies.sum(dim=1, keepdim=True) + self.num_classes * self.eps
        )
        return torch.log(probabilities)

    def forward(self, detector_energies: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(detector_energies))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
