from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
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

    def __init__(self):
        super().__init__()
        self.register_buffer('patch_weight', torch.zeros(1024, 3, 2, 16, 16, dtype=torch.bfloat16))
        self.register_buffer('patch_bias', torch.zeros(1024, dtype=torch.bfloat16))
        self.register_buffer('position', torch.zeros(784, 1024, dtype=torch.bfloat16))
        self.register_buffer('ready', torch.tensor(False))

    def forward(self, images):
        self.validate_frames(images)
        leading_shape = images.shape[:-3]
        temporal_pixels = self.temporal_pair(images)
        patch_grid = self.project_patches(temporal_pixels)
        grouped_tokens = self.group_patches(patch_grid)
        positioned_tokens = grouped_tokens + self.position
        pooled_tokens = self.pool_patches(positioned_tokens)
        return pooled_tokens.reshape(*leading_shape, 49, 1024).to(torch.float16)

    def validate_frames(self, images):
        if not bool(self.ready):
            raise RuntimeError('Visual frontend weights have not been loaded')
        if tuple(images.shape[-3:]) != (3, 448, 448) or not images.is_floating_point():
            raise ValueError('Expected normalized floating-point RGB frames shaped [...,3,448,448]')

    def temporal_pair(self, images):
        frames = images.reshape(-1, 3, 448, 448)
        paired_frames = frames.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        return paired_frames.to(self.patch_weight.dtype)

    def project_patches(self, temporal_pixels):
        embedded = F.conv3d(
            temporal_pixels,
            self.patch_weight,
            self.patch_bias,
            stride=(2, 16, 16),
        )
        return embedded[:, :, 0]

    def group_patches(self, patch_grid):
        groups = patch_grid.reshape(-1, 1024, 14, 2, 14, 2)
        groups = groups.permute(0, 2, 4, 3, 5, 1)
        return groups.reshape(-1, 784, 1024)

    def pool_patches(self, positioned_tokens):
        first_groups = positioned_tokens.reshape(-1, 196, 4, 1024)
        first_pooled = first_groups.mean(2)
        first_grid = first_pooled.reshape(-1, 14, 14, 1024)
        second_groups = first_grid.reshape(-1, 7, 2, 7, 2, 1024)
        return second_groups.mean((2, 4))


class Router(nn.Module):

    def __init__(self, config, token_count):
        super().__init__()
        self.temperature = config.temperature
        self.norm = nn.LayerNorm(config.width)
        self.fc1 = nn.Linear(token_count * config.width, config.router_hidden)
        self.fc2 = nn.Linear(config.router_hidden, 4)
        self.activation = nn.ReLU()

    def forward(self, tokens):
        normalized = self.norm(tokens)
        flattened = normalized.flatten(-2)
        hidden = self.activation(self.fc1(flattened))
        logits = self.fc2(hidden)
        probabilities = (logits / self.temperature).softmax(-1)
        indices = probabilities.topk(2, dim=-1).indices
        selected = torch.zeros_like(probabilities).scatter(-1, indices, 1.0)
        weights = probabilities * selected
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return {'weights': weights, 'probabilities': probabilities, 'selected': selected.bool()}


class Experts(nn.Module):

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

    def __init__(self, config):
        super().__init__()
        self.size = config.canvas
        self.pixel_pitch_m = config.pitch_um * 1e-6
        self.wavelength_m = config.wavelength_nm * 1e-9
        self.distance_m = config.distance_m
        self.cutoff_deg = config.cutoff_deg

        centered_phase = self.build_phase_kernel()
        self.register_buffer('phase', centered_phase, persistent=False)
        unshifted_transfer = torch.fft.ifftshift(centered_phase, dim=(-2, -1))
        self.register_buffer('transfer', unshifted_transfer, persistent=False)

    def frequency_grid(self):
        frequency = np.fft.fftfreq(self.size, d=self.pixel_pitch_m)
        centered = np.fft.fftshift(frequency)
        return np.meshgrid(centered, centered)

    def build_phase_kernel(self):
        frequency_x, frequency_y = self.frequency_grid()
        squared_frequency = frequency_x ** 2 + frequency_y ** 2
        argument = (2 * np.pi) ** 2 * (
            (1.0 / self.wavelength_m) ** 2 - frequency_x ** 2 - frequency_y ** 2
        )
        magnitude = np.sqrt(np.abs(argument))
        longitudinal_wave_number = np.where(argument >= 0, magnitude, 1j * magnitude)
        longitudinal_wave_number = torch.from_numpy(longitudinal_wave_number)
        phase = torch.exp(1j * longitudinal_wave_number * self.distance_m)

        radial_wave_number = 2 * np.pi * np.sqrt(squared_frequency)
        cutoff = (2 * np.pi / self.wavelength_m) * math.sin(math.radians(self.cutoff_deg))
        aperture = torch.from_numpy((argument >= 0) & (radial_wave_number <= cutoff))
        phase = phase.to(torch.complex64)
        return torch.where(aperture, phase, torch.zeros_like(phase))

    def forward(self, field):
        if field.ndim < 2 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(f'Expected fields with spatial shape [{self.size},{self.size}]')
        field = field.to(torch.complex64)
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        centered_spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        propagated_spectrum = centered_spectrum * self.phase
        unshifted_spectrum = torch.fft.ifftshift(propagated_spectrum, dim=(-2, -1))
        return torch.fft.ifft2(unshifted_spectrum, dim=(-2, -1))


class Global(nn.Module):

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

    @property
    def frame_offsets(self):
        return ((0, 0), (0, 59), (59, 0), (59, 59))

    def fields(self, tokens):
        value = F.relu(self.width_to_field(tokens))
        if self.is_vision:
            transposed = value.transpose(-2, -1)
            spatial_projection = self.tokens_to_field(transposed)
            value = F.relu(spatial_projection).transpose(-2, -1)
        else:
            missing_rows = 56 - value.shape[-2]
            value = F.pad(value, (0, 0, 0, missing_rows))
        mean_intensity = value.square().mean((-2, -1), keepdim=True)
        scale = mean_intensity.clamp_min(1e-12).sqrt()
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
        mean_intensity = patches.mean((-2, -1), keepdim=True)
        return patches / mean_intensity.clamp_min(1e-6)

    def input_field(self, tokens):
        fields = self.fields(tokens)
        canvas = fields.new_zeros(len(fields), self.cfg.canvas, self.cfg.canvas)
        for slot, (row, column) in enumerate(self.origins):
            if self.is_vision:
                for frame, (offset_y, offset_x) in enumerate(self.frame_offsets):
                    top = row + offset_y + 14
                    left = column + offset_x + 14
                    canvas[:, top:top+27, left:left+27] = fields[:, slot, frame]
            else:
                top = row + 29
                left = column + 29
                canvas[:, top:top+56, left:left+56] = fields[:, slot]
        return canvas

    def modulation_field(self, canvas):
        mask = torch.ones_like(canvas, dtype=torch.complex64)
        modulation = self.modulation()
        for slot, (row, column) in enumerate(self.origins):
            top = row + 2
            left = column + 2
            mask[:, top:top+111, left:left+111] = modulation[slot]
        return mask

    def detector_regions(self, detector):
        patches = []
        for row, column in self.origins:
            if self.is_vision:
                pieces = []
                for offset_y, offset_x in self.frame_offsets:
                    top = row + offset_y
                    left = column + offset_x
                    pieces.append(detector[:, top:top+56, left:left+56])
                patches.append(torch.stack(pieces, 1))
            else:
                patches.append(detector[:, row:row+115, column:column+115])
        patches = self.process_detector(torch.stack(patches, 1).clamp_min(0))
        return patches

    def detector_tokens(self, patches, count):
        leading = patches.shape[:-2]
        detector_images = patches.reshape(-1, 1, *patches.shape[-2:])
        pooled = F.adaptive_avg_pool2d(detector_images, (count, 96))[:, 0]
        normalized = self.norm(pooled)
        activated = F.relu(normalized)
        output = self.output(activated)
        return output.reshape(*leading, count, output.shape[-1])

    def forward(self, tokens):
        canvas = self.input_field(tokens)
        mask = self.modulation_field(canvas)
        modulated_field = canvas.to(torch.complex64) * mask
        output_field = self.propagation(modulated_field)
        detector = output_field.abs().square().float()
        patches = self.detector_regions(detector)
        count = self.cfg.tokens if self.is_vision else tokens.shape[-2]
        return self.detector_tokens(patches, count)


class LightGen2(nn.Module):

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
        self.vision_global = Global(config, 'vision')
        self.vision_merger = nn.Sequential(nn.LayerNorm(2 * config.width),
                                          nn.Linear(2 * config.width, config.width), nn.ReLU())
        self.vision_frame_position = nn.Parameter(torch.empty(1, 1, 4, config.width))
        self.language_sequence_position = nn.Parameter(torch.empty(1, 64, config.width))
        self.language_router = Router(config, 4 + config.prompt_length)
        self.language_experts = Experts(config)
        self.language_global = Global(config, 'language')
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
        visual_embeddings = self.vision_frontend(normalized_rgb)
        return self.predict_from_embeddings(visual_embeddings)

    @torch.inference_mode()
    def predict_from_embeddings(self, visual_embeddings):
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
        raise ValueError('Expected the LightGen-II inference checkpoint')
    config = dict(payload['config'])
    config.setdefault('shared_global_phase', False)
    if config['shared_global_phase'] != (payload['architecture'] == ARCHITECTURE):
        raise ValueError('Global phase sharing conflicts with the checkpoint architecture')
    model = LightGen2(ModelConfig(**config))
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload['metadata']
