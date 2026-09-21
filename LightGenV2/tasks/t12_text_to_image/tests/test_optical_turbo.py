from pathlib import Path

import torch
from torch import nn

from LightGenV2.tasks.t12_text_to_image.optical_turbo import (
    OpticalTurboConfig,
    ParallelOpticalMidBlock,
    load_optical_turbo_config,
    attach_parallel_optical_mid,
)


class _ElectronicRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input = None
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, temb, **kwargs):
        self.last_input = hidden_states.detach().clone()
        assert kwargs["encoder_hidden_states"] is not None
        return hidden_states * self.scale


def _config() -> OpticalTurboConfig:
    return OpticalTurboConfig(
        optical_width=8, grid=2, experts=2, top_k=1,
        alpha_initial=0.05, alpha_minimum=0.02, alpha_maximum=0.5,
        rms_epsilon=1e-6, batch_size=2, epochs=1,
        learning_rate=1e-4, phase_learning_rate=1e-3,
        weight_decay=0.0, detail_weight=0.0, num_workers=0,
    )


def test_parallel_mid_preserves_shape_and_freezes_electronic() -> None:
    electronic = _ElectronicRecorder()
    block = ParallelOpticalMidBlock(
        electronic, channels=4, timestep_dim=6, condition_dim=5, config=_config()
    )
    block.freeze_electronic()
    value = torch.randn(2, 4, 2, 2)
    output = block(
        value, torch.randn(2, 6), encoder_hidden_states=torch.randn(2, 3, 5)
    )
    assert output.shape == value.shape
    torch.testing.assert_close(electronic.last_input, value)
    assert not electronic.scale.requires_grad
    assert all(parameter.requires_grad for _, parameter in block.optical_parameters())


def test_optical_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "qwen_bksdm_v2_tiny_parallel_optical_v0.yaml"
    config = load_optical_turbo_config(path)
    assert config.optical_width == 224
    assert config.grid == 8
    assert config.top_k == 2


class _TimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_2 = nn.Linear(6, 6)


class _CompressedUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mid_block = None
        self.time_embedding = _TimeEmbedding()
        self.config = type("Config", (), {
            "block_out_channels": (2, 4),
            "cross_attention_dim": 5,
        })()


def test_attach_uses_identity_when_compressed_unet_removed_mid_block() -> None:
    unet = _CompressedUNet()
    wrapper = attach_parallel_optical_mid(unet, _config())
    assert unet.mid_block is wrapper
    assert wrapper.has_cross_attention is True
    assert wrapper.electronic_was_present is False
    assert "identity residual" in wrapper.architecture_report()["electronic_branch"]
