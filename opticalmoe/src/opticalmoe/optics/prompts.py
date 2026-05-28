import torch
import torch.nn as nn


class PromptModule(nn.Module):
    """Base class for physical prompt planes."""

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityPrompt(PromptModule):
    """Reserved prompt plane with no trainable modulation."""

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field
