"""Frozen inference model for the selected LightGen-II temporal checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'lightgen2_temporal_inference_v1'
QUALITY_WORDS = ('Bad', 'Poor', 'Fair', 'Good', 'Excellent')


@dataclass(frozen=True)
class ModelConfig:
    alpha_eval: float
    width: int = 192
    router_hidden: int = 128
    readout_hidden: int = 32
    prompt_length: int = 38
    videos: int = 16
    frames: int = 4
    tokens: int = 49
    temperature: float = 1.1
    wavelength_nm: float = 532.0
    pitch_um: float = 17.0
    distance_m: float = 0.1
    cutoff_deg: float = 0.5
    canvas: int = 518
    active: int = 478

    def __post_init__(self):
        geometry = (self.width, self.router_hidden, self.readout_hidden,
                    self.prompt_length, self.videos, self.frames,
                    self.tokens, self.canvas, self.active)
        if geometry != (192, 128, 32, 38, 16, 4, 49, 518, 478):
            raise ValueError('Checkpoint dimensions differ from the selected architecture')
        positive = (self.temperature, self.wavelength_nm, self.pitch_um,
                    self.distance_m, self.cutoff_deg)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('Optical and routing scales must be finite and positive')
        if not math.isfinite(self.alpha_eval) or not 0 <= self.alpha_eval <= 1:
            raise ValueError('Invalid modulation coefficient')


class FrozenFront(nn.Module):
    """Embedded patch projection and position tensors for four RGB frames."""

    def __init__(self):
        super().__init__()
        self.register_buffer('patch_weight', torch.zeros(1024, 3, 2, 16, 16, dtype=torch.bfloat16))
        self.register_buffer('patch_bias', torch.zeros(1024, dtype=torch.bfloat16))
        self.register_buffer('position', torch.zeros(784, 1024, dtype=torch.bfloat16))
        self.register_buffer('ready', torch.tensor(False))

    def forward(self, images):
        if not bool(self.ready):
            raise RuntimeError('Visual frontend weights have not been loaded')
        if tuple(images.shape[-3:]) != (3, 448, 448) or images.dtype != torch.uint8:
            raise ValueError('Expected uint8 RGB frames shaped [...,3,448,448]')
        leading = images.shape[:-3]
        rgb = (images.reshape(-1, 3, 448, 448).float() / 255.0 - 0.5) / 0.5
        rgb = rgb.unsqueeze(2).expand(-1, -1, 2, -1, -1).to(self.patch_weight.dtype)
        hidden = F.conv3d(rgb, self.patch_weight, self.patch_bias, stride=(2, 16, 16))[:, :, 0]
        hidden = hidden.reshape(-1, 1024, 14, 2, 14, 2)
        hidden = hidden.permute(0, 2, 4, 3, 5, 1).reshape(-1, 784, 1024)
        hidden = hidden + self.position
        merged = hidden.reshape(-1, 196, 4, 1024).mean(2).reshape(-1, 14, 14, 1024)
        pooled = merged.reshape(-1, 7, 2, 7, 2, 1024).mean((2, 4))
        return pooled.reshape(*leading, 49, 1024).to(torch.float16)


class Router(nn.Module):
    """One shared Top2 decision for all tokens in each input item."""

    def __init__(self, config, token_count):
        super().__init__()
        self.temperature = config.temperature
        self.norm = nn.LayerNorm(config.width)
        self.fc1 = nn.Linear(token_count * config.width, config.router_hidden)
        self.fc2 = nn.Linear(config.router_hidden, 4)
        self.activation = nn.ReLU()

    def forward(self, tokens):
        logits = self.fc2(self.activation(self.fc1(self.norm(tokens).flatten(-2))))
        probabilities = (logits / self.temperature).softmax(-1)
        indices = probabilities.topk(2, dim=-1).indices
        selected = torch.zeros_like(probabilities).scatter(-1, indices, 1.0)
        weights = probabilities * selected
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return {'weights': weights, 'probabilities': probabilities, 'selected': selected.bool()}


class Experts(nn.Module):
    """Four independent tokenwise linear matrices with shared frame parameters."""

    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(config.width, config.width, bias=False) for _ in range(4)
        ])

    def forward(self, tokens, weights):
        output = torch.zeros_like(tokens)
        for index, expert in enumerate(self.experts):
            output = output + expert(tokens) * weights[..., index, None, None]
        return output


class Propagation(nn.Module):
    """Whole-canvas coherent angular-spectrum propagation."""

    def __init__(self, config):
        super().__init__()
        frequency = torch.fft.fftfreq(config.canvas, d=config.pitch_um * 1e-6, dtype=torch.float64)
        frequency_y, frequency_x = torch.meshgrid(frequency, frequency, indexing='ij')
        wavelength = config.wavelength_nm * 1e-9
        squared_frequency = frequency_x.square() + frequency_y.square()
        argument = (2 * math.pi) ** 2 * (
            (1 / wavelength) ** 2 - frequency_x.square() - frequency_y.square())
        radial_frequency = 2 * math.pi * torch.sqrt(squared_frequency)
        cutoff = (2 * math.pi / wavelength) * math.sin(math.radians(config.cutoff_deg))
        valid = (argument >= 0) & (radial_frequency <= cutoff)
        transfer = torch.exp(1j * config.distance_m * torch.sqrt(argument.clamp_min(0)))
        transfer = transfer.to(torch.complex64)
        self.register_buffer('transfer', torch.where(valid, transfer, torch.zeros_like(transfer)),
                             persistent=False)

    def forward(self, field):
        spectrum = torch.fft.fft2(field.to(torch.complex64))
        return torch.fft.ifft2(spectrum * self.transfer)


class GlobalStage(nn.Module):
    """Amplitude encoding, phase modulation, propagation and detector readout."""

    def __init__(self, config, frame_level):
        super().__init__()
        self.cfg = config
        self.frame_level = frame_level
        side = 27 if frame_level else 56
        self.width_to_field = nn.Linear(config.width, side)
        if frame_level:
            self.tokens_to_field = nn.Linear(config.tokens, side)
        self.raw_global_phase = nn.Parameter(torch.empty(16, 111, 111))
        self.propagation = Propagation(config)
        self.norm = nn.LayerNorm(96)
        self.output = nn.Linear(96, config.width) if frame_level else nn.Identity()

    @property
    def origins(self):
        return [(23 + 119 * row, 23 + 119 * column)
                for row in range(4) for column in range(4)]

    def fields(self, tokens):
        value = F.relu(self.width_to_field(tokens))
        if self.frame_level:
            value = F.relu(self.tokens_to_field(value.transpose(-2, -1))).transpose(-2, -1)
        else:
            value = F.pad(value, (0, 0, 0, 56 - value.shape[-2]))
        scale = value.square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        return value / scale

    def phase(self, *, wrapped=False):
        value = 2 * math.pi * torch.sigmoid(self.raw_global_phase)
        return value.remainder(2 * math.pi) if wrapped else value

    def modulation(self):
        value = torch.exp(1j * self.phase())
        coefficient = value.real.new_tensor(self.cfg.alpha_eval)
        return coefficient * value + (1 - coefficient)

    def process_detector(self, patches):
        return patches / patches.mean((-2, -1), keepdim=True).clamp_min(1e-6)

    def forward(self, tokens):
        fields = self.fields(tokens)
        canvas = fields.new_zeros(len(fields), 518, 518)
        frame_offsets = [(0, 0), (0, 59), (59, 0), (59, 59)]
        for slot, (row, column) in enumerate(self.origins):
            if self.frame_level:
                for frame, (dy, dx) in enumerate(frame_offsets):
                    canvas[:, row+dy+14:row+dy+41, column+dx+14:column+dx+41] = fields[:, slot, frame]
            else:
                canvas[:, row+29:row+85, column+29:column+85] = fields[:, slot]
        mask = torch.ones_like(canvas, dtype=torch.complex64)
        modulation = self.modulation()
        for slot, (row, column) in enumerate(self.origins):
            mask[:, row+2:row+113, column+2:column+113] = modulation[slot]
        detector = self.propagation(canvas.to(torch.complex64) * mask).abs().square().float()
        patches = []
        for row, column in self.origins:
            if self.frame_level:
                pieces = [detector[:, row+dy:row+dy+56, column+dx:column+dx+56]
                          for dy, dx in frame_offsets]
                patches.append(torch.stack(pieces, 1))
            else:
                patches.append(detector[:, row:row+115, column:column+115])
        patches = self.process_detector(torch.stack(patches, 1).clamp_min(0))
        leading = patches.shape[:-2]
        count = self.cfg.tokens if self.frame_level else tokens.shape[-2]
        pooled = F.adaptive_avg_pool2d(patches.reshape(-1, 1, *patches.shape[-2:]), (count, 96))[:, 0]
        output = self.output(F.relu(self.norm(pooled)))
        return output.reshape(*leading, count, output.shape[-1])


class LightGenTemporal(nn.Module):
    """Selected serial visual and vision-language inference graph."""

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.front = FrozenFront()
        self.register_buffer('prompt_embedding', torch.zeros(config.prompt_length, 2048, dtype=torch.float16))
        self.register_buffer('prompt_ready', torch.tensor(False))
        self.vision_adapter = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, config.width))
        self.language_adapter = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, config.width))
        self.frame_router = Router(config, 49)
        self.frame_experts = Experts(config)
        self.frame_global = GlobalStage(config, True)
        self.frame_merger = nn.Sequential(nn.LayerNorm(2 * config.width),
                                          nn.Linear(2 * config.width, config.width), nn.ReLU())
        self.frame_position = nn.Parameter(torch.empty(1, 1, 4, config.width))
        self.sequence_position = nn.Parameter(torch.empty(1, 64, config.width))
        self.video_router = Router(config, 4 + config.prompt_length)
        self.video_experts = Experts(config)
        self.video_global = GlobalStage(config, False)
        self.readout = nn.Sequential(nn.Flatten(-2),
                                     nn.Linear((4 + config.prompt_length) * 96, config.readout_hidden),
                                     nn.ReLU(), nn.Linear(config.readout_hidden, 5))
        self.register_buffer('level_scores', torch.zeros(5))
        self.register_buffer('quality_ready', torch.tensor(False))

    @torch.no_grad()
    def set_prompt(self, embedding):
        value = embedding.reshape(-1, 2048)
        if value.shape != self.prompt_embedding.shape:
            raise ValueError('Fixed prompt shape mismatch')
        self.prompt_embedding.copy_(value.to(self.prompt_embedding))
        self.prompt_ready.fill_(True)

    def encode_frames(self, vision_tokens):
        if tuple(vision_tokens.shape[1:]) != (16, 4, 49, 1024):
            raise ValueError('Expected [batch,16,4,49,1024] visual features')
        vision = self.vision_adapter(vision_tokens.float())
        routing = self.frame_router(vision)
        vision = self.frame_experts(vision, routing['weights'])
        vision = self.frame_global(vision)
        summary = torch.cat((vision.mean(-2), vision.amax(-2)), -1)
        frames = self.frame_merger(summary) + self.frame_position
        return frames, routing

    @torch.inference_mode()
    def forward(self, vision_tokens=None, *, pixels=None):
        if (vision_tokens is None) == (pixels is None):
            raise ValueError('Provide exactly one visual-feature or pixel input')
        if pixels is not None:
            vision_tokens = self.front(pixels)
        if not bool(self.prompt_ready) or not bool(self.quality_ready):
            raise RuntimeError('Prompt or grade-score tensors have not been loaded')
        frames, frame_routing = self.encode_frames(vision_tokens)
        prompt = self.language_adapter(self.prompt_embedding.float())
        prompt = prompt[None, None].expand(len(frames), 16, -1, -1)
        sequence = torch.cat((frames, prompt), dim=-2)
        sequence = sequence + self.sequence_position[:, None, :sequence.shape[-2]]
        video_routing = self.video_router(sequence)
        sequence = self.video_experts(sequence, video_routing['weights'])
        sequence = self.video_global(sequence)
        logits = self.readout(sequence)
        probabilities = logits.softmax(-1)
        prediction = probabilities @ self.level_scores
        return {'prediction': prediction, 'quality_logits': logits,
                'quality_probabilities': probabilities,
                'routing': {'frame': frame_routing, 'video': video_routing}}


def load_lightgen(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != 1 or payload['architecture'] != ARCHITECTURE:
        raise ValueError('Expected the LightGen2 inference checkpoint')
    model = LightGenTemporal(ModelConfig(**payload['config']))
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload['metadata']
