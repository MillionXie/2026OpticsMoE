from pathlib import Path

import torch

from LightGenV2.tasks.t12_text_to_image.compact_turbo import (
    latent_gradient_loss,
    load_compact_turbo_config,
    one_step_denoise,
)


class _ZeroUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, value, timestep, encoder_hidden_states, return_dict=False):
        self.calls += 1
        assert timestep.shape == (len(value),)
        assert encoder_hidden_states.shape[0] == len(value)
        assert return_dict is False
        return (torch.zeros_like(value),)


def test_one_step_operator_calls_unet_once() -> None:
    model = _ZeroUNet()
    noise = torch.randn(2, 4, 8, 8)
    condition = torch.randn(2, 5, 6)
    sigma = torch.tensor(3.0)
    output = one_step_denoise(model, noise, condition, sigma)
    torch.testing.assert_close(output, noise * sigma)
    assert model.calls == 1


def test_gradient_loss_is_zero_for_identical_latents() -> None:
    value = torch.randn(2, 4, 8, 8)
    assert float(latent_gradient_loss(value, value)) == 0.0


def test_compact_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "qwen_bksdm_v2_tiny_one_step.yaml"
    config = load_compact_turbo_config(path)
    assert config.batch_size == 32
    assert config.epochs == 4
