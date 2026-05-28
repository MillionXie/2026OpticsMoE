import torch
import torch.nn as nn


class ElectronicReadout(nn.Module):
    """Electronic classifier head operating on detector energies."""

    def __init__(
        self,
        num_classes: int,
        readout_type: str = "optical_only",
        logit_scale: float = 10.0,
        hidden_dim: int = 64,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.readout_type = readout_type
        self.logit_scale = float(logit_scale)

        if readout_type == "optical_only":
            self.net = nn.Identity()
        elif readout_type == "linear":
            self.net = nn.Linear(num_classes, num_classes)
        elif readout_type == "mlp":
            act = self._activation(activation)
            self.net = nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                act,
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported readout type: {readout_type}")

    def _activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported readout activation: {name}")

    def forward(self, detector_energies: torch.Tensor) -> torch.Tensor:
        if self.readout_type == "optical_only":
            return detector_energies * self.logit_scale
        return self.net(detector_energies)
