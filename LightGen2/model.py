"""LightGen2 inference with vision and language optical branches."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'lightgen2_temporal_inference_v3_shared_phase'
LEGACY_ARCHITECTURE = 'lightgen2_temporal_inference_v2'
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
    shared_global_phase: bool = True

    def __post_init__(self):
        if not isinstance(self.shared_global_phase, bool):
            raise ValueError('Global phase sharing must be a boolean')
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


class FrozenVisionFront(nn.Module):
    """Saved Qwen patch and position tensors followed by fixed mean pooling.

    The constructor allocates buffers. The LightGen2 checkpoint supplies their
    frozen values through load_state_dict before inference.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('patch_weight', torch.zeros(1024, 3, 2, 16, 16, dtype=torch.bfloat16))
        self.register_buffer('patch_bias', torch.zeros(1024, dtype=torch.bfloat16))
        self.register_buffer('position', torch.zeros(784, 1024, dtype=torch.bfloat16))
        self.register_buffer('ready', torch.tensor(False))

    def forward(self, images):
        if not bool(self.ready):
            raise RuntimeError('Visual frontend weights have not been loaded')
        if tuple(images.shape[-3:]) != (3, 448, 448) or not images.is_floating_point():
            raise ValueError('Expected normalized floating-point RGB frames shaped [...,3,448,448]')
        leading = images.shape[:-3]
        rgb = images.reshape(-1, 3, 448, 448)
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
    """Four tokenwise linear matrices shared across items in one branch."""

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


class OpticalGlobal(nn.Module):
    """Amplitude encoding, phase modulation, propagation and detector readout."""

    def __init__(self, config, branch):
        super().__init__()
        if branch not in ('vision', 'language'):
            raise ValueError('Expected the vision or language branch')
        self.cfg = config
        self.branch = branch
        self.is_vision = branch == 'vision'
        side = 27 if self.is_vision else 56
        self.width_to_field = nn.Linear(config.width, side)
        if self.is_vision:
            self.tokens_to_field = nn.Linear(config.tokens, side)
        phase_count = 1 if config.shared_global_phase else config.videos
        self.raw_phase = nn.Parameter(torch.empty(phase_count, 111, 111))
        self.propagation = Propagation(config)
        self.norm = nn.LayerNorm(96)
        self.output = nn.Linear(96, config.width) if self.is_vision else nn.Identity()

    @property
    def origins(self):
        return [(23 + 119 * row, 23 + 119 * column)
                for row in range(4) for column in range(4)]

    def fields(self, tokens):
        value = F.relu(self.width_to_field(tokens))
        if self.is_vision:
            value = F.relu(self.tokens_to_field(value.transpose(-2, -1))).transpose(-2, -1)
        else:
            value = F.pad(value, (0, 0, 0, 56 - value.shape[-2]))
        scale = value.square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        return value / scale

    def phase(self, *, wrapped=False):
        value = 2 * math.pi * torch.sigmoid(self.raw_phase)
        value = value.expand(self.cfg.videos, -1, -1)
        return value.remainder(2 * math.pi) if wrapped else value

    def modulation(self):
        value = torch.exp(1j * self.phase())
        coefficient = value.real.new_tensor(self.cfg.alpha_eval)
        return coefficient * value + (1 - coefficient)

    def process_detector(self, patches):
        return patches / patches.mean((-2, -1), keepdim=True).clamp_min(1e-6)

    def input_field(self, tokens):
        """Pack all 16 videos into the incident 518 by 518 amplitude field."""
        fields = self.fields(tokens)
        canvas = fields.new_zeros(len(fields), 518, 518)
        frame_offsets = [(0, 0), (0, 59), (59, 0), (59, 59)]
        for slot, (row, column) in enumerate(self.origins):
            if self.is_vision:
                for frame, (dy, dx) in enumerate(frame_offsets):
                    canvas[:, row+dy+14:row+dy+41, column+dx+14:column+dx+41] = fields[:, slot, frame]
            else:
                canvas[:, row+29:row+85, column+29:column+85] = fields[:, slot]
        return canvas

    def modulation_field(self, canvas):
        """Embed all 16 learned modulation tiles at their physical positions."""
        mask = torch.ones_like(canvas, dtype=torch.complex64)
        modulation = self.modulation()
        for slot, (row, column) in enumerate(self.origins):
            mask[:, row+2:row+113, column+2:column+113] = modulation[slot]
        return mask

    def forward(self, tokens):
        canvas = self.input_field(tokens)
        mask = self.modulation_field(canvas)
        detector = self.propagation(canvas.to(torch.complex64) * mask).abs().square().float()
        patches = []
        frame_offsets = [(0, 0), (0, 59), (59, 0), (59, 59)]
        for row, column in self.origins:
            if self.is_vision:
                pieces = [detector[:, row+dy:row+dy+56, column+dx:column+dx+56]
                          for dy, dx in frame_offsets]
                patches.append(torch.stack(pieces, 1))
            else:
                patches.append(detector[:, row:row+115, column:column+115])
        patches = self.process_detector(torch.stack(patches, 1).clamp_min(0))
        leading = patches.shape[:-2]
        count = self.cfg.tokens if self.is_vision else tokens.shape[-2]
        pooled = F.adaptive_avg_pool2d(patches.reshape(-1, 1, *patches.shape[-2:]), (count, 96))[:, 0]
        output = self.output(F.relu(self.norm(pooled)))
        return output.reshape(*leading, count, output.shape[-1])


class LightGen2(nn.Module):
    """Vision processing followed by language-conditioned quality readout.

    The language branch combines four visual summary tokens with 38 prompt
    tokens. Its router and experts process the complete 42-token sequence.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.vision_frontend = FrozenVisionFront()
        self.register_buffer('language_prompt_embedding', torch.zeros(config.prompt_length, 2048, dtype=torch.float16))
        self.register_buffer('language_prompt_ready', torch.tensor(False))
        self.vision_adapter = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, config.width))
        self.language_adapter = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, config.width))
        self.vision_router = Router(config, 49)
        self.vision_experts = Experts(config)
        self.vision_global = OpticalGlobal(config, 'vision')
        self.vision_merger = nn.Sequential(nn.LayerNorm(2 * config.width),
                                          nn.Linear(2 * config.width, config.width), nn.ReLU())
        self.vision_frame_position = nn.Parameter(torch.empty(1, 1, 4, config.width))
        self.language_sequence_position = nn.Parameter(torch.empty(1, 64, config.width))
        self.language_router = Router(config, 4 + config.prompt_length)
        self.language_experts = Experts(config)
        self.language_global = OpticalGlobal(config, 'language')
        self.readout = nn.Sequential(nn.Flatten(-2),
                                     nn.Linear((4 + config.prompt_length) * 96, config.readout_hidden),
                                     nn.ReLU(), nn.Linear(config.readout_hidden, 5))
        self.register_buffer('grade_scores', torch.zeros(5))
        self.register_buffer('readout_ready', torch.tensor(False))

    @torch.no_grad()
    def set_prompt(self, embedding):
        value = embedding.reshape(-1, 2048)
        if value.shape != self.language_prompt_embedding.shape:
            raise ValueError('Fixed prompt shape mismatch')
        self.language_prompt_embedding.copy_(value.to(self.language_prompt_embedding))
        self.language_prompt_ready.fill_(True)

    def encode_vision(self, visual_embeddings):
        if tuple(visual_embeddings.shape[1:]) != (16, 4, 49, 1024):
            raise ValueError('Expected [batch,16,4,49,1024] visual features')
        vision = self.vision_adapter(visual_embeddings.float())
        routing = self.vision_router(vision)
        vision = self.vision_experts(vision, routing['weights'])
        vision = self.vision_global(vision)
        summary = torch.cat((vision.mean(-2), vision.amax(-2)), -1)
        visual_summary = self.vision_merger(summary) + self.vision_frame_position
        return visual_summary, routing

    def encode_language(self, visual_summary):
        prompt = self.language_adapter(self.language_prompt_embedding.float())
        prompt = prompt[None, None].expand(len(visual_summary), 16, -1, -1)
        sequence = torch.cat((visual_summary, prompt), dim=-2)
        sequence = sequence + self.language_sequence_position[:, None, :sequence.shape[-2]]
        routing = self.language_router(sequence)
        sequence = self.language_experts(sequence, routing['weights'])
        sequence = self.language_global(sequence)
        return sequence, routing

    @torch.inference_mode()
    def forward(self, normalized_rgb):
        """Predict from normalized RGB shaped [batch,16,4,3,448,448]."""
        visual_embeddings = self.vision_frontend(normalized_rgb)
        return self.predict_from_embeddings(visual_embeddings)

    @torch.inference_mode()
    def predict_from_embeddings(self, visual_embeddings):
        """Predict from saved frontend outputs shaped [batch,16,4,49,1024]."""
        if not bool(self.language_prompt_ready) or not bool(self.readout_ready):
            raise RuntimeError('Prompt or grade-score tensors have not been loaded')
        visual_summary, vision_routing = self.encode_vision(visual_embeddings)
        sequence, language_routing = self.encode_language(visual_summary)
        logits = self.readout(sequence)
        probabilities = logits.softmax(-1)
        prediction = probabilities @ self.grade_scores
        return {'prediction': prediction, 'quality_logits': logits,
                'quality_probabilities': probabilities,
                'routing': {'vision': vision_routing, 'language': language_routing}}


def load_lightgen2(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != 2 or payload['architecture'] not in (ARCHITECTURE, LEGACY_ARCHITECTURE):
        raise ValueError('Expected the LightGen2 inference checkpoint')
    config = dict(payload['config'])
    config.setdefault('shared_global_phase', False)
    if config['shared_global_phase'] != (payload['architecture'] == ARCHITECTURE):
        raise ValueError('Global phase sharing conflicts with the checkpoint architecture')
    model = LightGen2(ModelConfig(**config))
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload['metadata']
