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
        input_norm: str = "none",
        norm_affine: bool = True,
        hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.readout_type = readout_type
        self.logit_scale = float(logit_scale)
        self.input_norm = input_norm
        self.norm_affine = bool(norm_affine)
        self.hidden_layers = int(hidden_layers)
        self.dropout = float(dropout)

        norm = self._normalization(input_norm, num_classes, self.norm_affine)
        if readout_type == "optical_only":
            self.net = norm
        elif readout_type == "linear":
            self.net = nn.Sequential(
                norm,
                nn.Linear(num_classes, num_classes),
            )
        elif readout_type == "mlp":
            if self.hidden_layers <= 0:
                raise ValueError("hidden_layers must be positive for mlp readout.")
            layers = [norm, nn.Linear(num_classes, hidden_dim)]
            for _index in range(self.hidden_layers):
                layers.append(self._activation(activation))
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
                if _index < self.hidden_layers - 1:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported readout type: {readout_type}")

    def _normalization(
        self,
        name: str,
        num_classes: int,
        affine: bool,
    ) -> nn.Module:
        if name in {"none", None}:
            return nn.Identity()
        if name == "layernorm":
            return nn.LayerNorm(num_classes, elementwise_affine=affine)
        raise ValueError(f"Unsupported readout input_norm: {name}")

    def _activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported readout activation: {name}")

    def forward(self, detector_energies: torch.Tensor) -> torch.Tensor:
        if self.readout_type == "optical_only":
            return self.net(detector_energies) * self.logit_scale
        return self.net(detector_energies)
