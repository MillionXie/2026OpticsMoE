import torch
from torch import nn

from LightGenV2.tasks.t12_text_to_image.pruned_turbo import (
    AttentionBypass,
    apply_attention_pruning,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([_Attention(), _Attention()])


class _UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([_Block()])
        self.up_blocks = nn.ModuleList([_Block()])


def test_attention_bypass_preserves_tensor_and_diffusers_tuple_contract() -> None:
    value = torch.randn(2, 4, 3, 3)
    block = AttentionBypass()
    assert block(value, return_dict=False)[0] is value
    assert block(value).sample is value


def test_pruning_removes_only_selected_attention_parameters() -> None:
    unet = _UNet()
    before = sum(parameter.numel() for parameter in unet.parameters())
    report = apply_attention_pruning(
        unet, ({"side": "up", "block": 0, "attention": 1},)
    )
    assert isinstance(unet.up_blocks[0].attentions[1], AttentionBypass)
    assert isinstance(unet.up_blocks[0].attentions[0], _Attention)
    assert report["removed_parameters"] == 3
    assert report["remaining_parameters"] == before - 3
