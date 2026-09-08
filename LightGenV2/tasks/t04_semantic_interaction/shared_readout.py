"""One identical two-convolution-group task readout for optical and Qwen models."""
import hashlib
import torch
from torch import nn
from .embedding_model import PositionReadout
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.modeling import SemanticGridDecoder
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.modeling import ConditionedResidual2D


class SharedGridReadout(nn.Module):
    contract = 'positionlinear64_width192_film01_coord_dwconv_dilation1_2_grid6_v1'

    def __init__(self, width=192, max_tokens=64):
        super().__init__()
        self.position_readout = PositionReadout(max_tokens)
        self.language_pool = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.post_film = nn.Linear(width, width * 2)
        nn.init.zeros_(self.post_film.weight)
        nn.init.zeros_(self.post_film.bias)
        self.coordinate_projection = nn.Conv2d(2, width, 1)
        self.editor = nn.ModuleList([ConditionedResidual2D(width, d) for d in (1, 2)])
        self.decoder = SemanticGridDecoder(width, 6, 16)
        self.task_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))

    def summarize(self, language_groups):
        return self.language_pool(self.position_readout(language_groups))

    def forward(self, spatial, condition):
        if spatial.shape[1:] != (192, 14, 14) or condition.shape != (len(spatial), 192):
            raise ValueError('Shared head expects [B,192,14,14] and [B,192]')
        gamma, beta = self.post_film(condition).chunk(2, -1)
        spatial = spatial * (1 + .1 * torch.tanh(gamma)[:, :, None, None]) + .1 * torch.tanh(beta)[:, :, None, None]
        axis = torch.linspace(-1., 1., 14, device=spatial.device, dtype=spatial.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        spatial = spatial + self.coordinate_projection(torch.stack((xx, yy))[None])
        for layer in self.editor:
            spatial = layer(spatial, condition)
        category, edit = self.decoder(spatial)
        return {'category_logits': category, 'edit_logits': edit, 'task_logits': self.task_head(condition),
                'condition': condition, 'spatial': spatial}


def create_shared_readout(settings):
    # Both methods start with bit-identical readout parameters, independent of backbone RNG usage.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings.seed + 1000)
        head = SharedGridReadout(settings.electronic_width, settings.max_language_tokens)
    return head


def head_signature(head):
    digest = hashlib.sha256()
    for key, value in head.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().float().contiguous().numpy().tobytes())
    return {'contract': head.contract, 'initial_sha256': digest.hexdigest(),
            'parameters': sum(p.numel() for p in head.parameters())}
