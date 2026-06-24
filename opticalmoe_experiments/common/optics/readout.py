import torch
import torch.nn as nn


class ElectronicReadout(nn.Module):
    def __init__(
        self,
        num_classes: int,
        readout_type: str = "mlp",
        logit_scale: float = 10.0,
        hidden_dim: int = 64,
        activation: str = "relu",
        input_norm: str = "layernorm",
        norm_affine: bool = True,
        hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.readout_type = readout_type
        self.logit_scale = float(logit_scale)
        norm = self._norm(input_norm, num_classes, norm_affine)
        if readout_type == "optical_only":
            self.net = norm
        elif readout_type == "linear":
            self.net = nn.Sequential(norm, nn.Linear(num_classes, num_classes))
        elif readout_type == "mlp":
            layers = [norm, nn.Linear(num_classes, hidden_dim)]
            for index in range(int(hidden_layers)):
                layers.append(self._activation(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                if index < int(hidden_layers) - 1:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported readout type: {readout_type}")

    def _norm(self, name, num_classes, affine):
        if name in {None, "none"}:
            return nn.Identity()
        if name == "layernorm":
            return nn.LayerNorm(num_classes, elementwise_affine=bool(affine))
        raise ValueError(f"Unsupported readout input_norm: {name}")

    def _activation(self, name):
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, detector_energies: torch.Tensor) -> torch.Tensor:
        logits = self.net(detector_energies)
        return logits * self.logit_scale if self.readout_type == "optical_only" else logits

