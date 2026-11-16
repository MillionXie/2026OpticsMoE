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


@dataclass(frozen=True)
class VisionFrontGeometry:
    image_side: int = 448
    channels: int = 3
    temporal_patch: int = 2
    patch_side: int = 16
    hidden_size: int = 1024
    merge_side: int = 2
    pool_side: int = 2

    def __post_init__(self):
        sizes = (self.image_side, self.channels, self.temporal_patch,
                 self.patch_side, self.hidden_size, self.merge_side, self.pool_side)
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError('Frontend dimensions must be positive integers')
        if self.image_side % self.patch_side:
            raise ValueError('The image grid must contain complete patches')
        if self.patch_grid % self.merge_side or self.merged_grid % self.pool_side:
            raise ValueError('The patch grid must contain complete spatial groups')

    @property
    def frame_shape(self):
        return (self.channels, self.image_side, self.image_side)

    @property
    def kernel_shape(self):
        return (self.hidden_size, self.channels, self.temporal_patch,
                self.patch_side, self.patch_side)

    @property
    def patch_grid(self):
        return self.image_side // self.patch_side

    @property
    def patch_tokens(self):
        return self.patch_grid * self.patch_grid

    @property
    def merged_grid(self):
        return self.patch_grid // self.merge_side

    @property
    def output_grid(self):
        return self.merged_grid // self.pool_side

    @property
    def output_tokens(self):
        return self.output_grid * self.output_grid


@dataclass(frozen=True)
class VisionFrameBatch:
    leading_shape: tuple
    geometry: VisionFrontGeometry

    @classmethod
    def from_images(cls, images, geometry):
        if not isinstance(images, torch.Tensor):
            raise ValueError('Qwen frontend input must be a tensor')
        if images.ndim < 3 or tuple(images.shape[-3:]) != geometry.frame_shape:
            raise ValueError('Expected normalized floating-point RGB frames shaped [...,3,448,448]')
        if not images.is_floating_point():
            raise ValueError('Expected normalized floating-point RGB frames shaped [...,3,448,448]')
        leading_shape = tuple(images.shape[:-3])
        if any(size <= 0 for size in leading_shape):
            raise ValueError('The frontend batch must contain at least one frame')
        return cls(leading_shape, geometry)

    @property
    def frame_count(self):
        return math.prod(self.leading_shape)

    @property
    def image_shape(self):
        return (*self.leading_shape, *self.geometry.frame_shape)

    @property
    def flattened_shape(self):
        return (self.frame_count, *self.geometry.frame_shape)

    @property
    def token_shape(self):
        return (self.frame_count, self.geometry.output_tokens,
                self.geometry.hidden_size)

    def flatten_frames(self, images):
        if tuple(images.shape) != self.image_shape:
            raise ValueError('Video and frame dimensions changed before patch projection')
        flattened_frames = images.reshape(*self.flattened_shape)
        return flattened_frames

    def restore_tokens(self, pooled_grid):
        expected = (self.frame_count, self.geometry.output_grid,
                    self.geometry.output_grid, self.geometry.hidden_size)
        if tuple(pooled_grid.shape) != expected:
            raise ValueError('The pooled Qwen grid does not match the input frame batch')
        flat_tokens = pooled_grid.reshape(*self.token_shape)
        restored_shape = (*self.leading_shape, self.geometry.output_tokens,
                          self.geometry.hidden_size)
        restored_tokens = flat_tokens.reshape(*restored_shape)
        return restored_tokens.to(torch.float16)


class QwenSpatialLayout(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry

    def validate_projection_grid(self, patch_grid):
        expected = (self.geometry.hidden_size, self.geometry.patch_grid,
                    self.geometry.patch_grid)
        if patch_grid.ndim != 4 or tuple(patch_grid.shape[1:]) != expected:
            raise ValueError('Projected Qwen patches require a channel-first spatial grid')

    def split_projection_grid(self, patch_grid):
        self.validate_projection_grid(patch_grid)
        frames = patch_grid.shape[0]
        grid = self.geometry.merged_grid
        group = self.geometry.merge_side
        width = self.geometry.hidden_size
        spatial_groups = patch_grid.reshape(frames, width, grid, group, grid, group)
        return spatial_groups

    def order_projection_groups(self, spatial_groups):
        ordered_groups = spatial_groups.permute(0, 2, 4, 3, 5, 1)
        return ordered_groups

    def flatten_projection_groups(self, ordered_groups):
        frames = ordered_groups.shape[0]
        width = self.geometry.hidden_size
        grouped_tokens = ordered_groups.reshape(frames, self.geometry.patch_tokens, width)
        return grouped_tokens

    def group_projection(self, patch_grid):
        spatial_groups = self.split_projection_grid(patch_grid)
        ordered_groups = self.order_projection_groups(spatial_groups)
        grouped_tokens = self.flatten_projection_groups(ordered_groups)
        return grouped_tokens

    def split_position_grid(self, spatial_positions):
        expected = (self.geometry.patch_tokens, self.geometry.hidden_size)
        if tuple(spatial_positions.shape) != expected:
            raise ValueError('Interpolated Qwen positions do not match the projected patch grid')
        grid = self.geometry.merged_grid
        group = self.geometry.merge_side
        width = self.geometry.hidden_size
        position_grid = spatial_positions.reshape(1, grid, group, grid, group, width)
        return position_grid

    def order_position_groups(self, position_grid):
        ordered_positions = position_grid.permute(0, 1, 3, 2, 4, 5)
        return ordered_positions

    def flatten_position_groups(self, ordered_positions):
        grouped_positions = ordered_positions.flatten(0, 4).contiguous()
        return grouped_positions

    def group_positions(self, spatial_positions):
        position_grid = self.split_position_grid(spatial_positions)
        ordered_positions = self.order_position_groups(position_grid)
        grouped_positions = self.flatten_position_groups(ordered_positions)
        return grouped_positions


class QwenPatchProjection(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.spatial_layout = QwenSpatialLayout(geometry)

    def validate_projection(self, weight, bias):
        if tuple(weight.shape) != self.geometry.kernel_shape:
            raise ValueError('Qwen patch-projection kernel dimensions differ')
        if tuple(bias.shape) != (self.geometry.hidden_size,):
            raise ValueError('Qwen patch-projection bias dimensions differ')
        if weight.device != bias.device or weight.dtype != bias.dtype:
            raise ValueError('Qwen patch-projection tensors require a common device and dtype')
        if not weight.is_floating_point() or not bias.is_floating_point():
            raise ValueError('Qwen patch-projection tensors must be floating point')

    def temporal_pair(self, images, dtype):
        frame_batch = VisionFrameBatch.from_images(images, self.geometry)
        frames = frame_batch.flatten_frames(images)
        temporal_frames = frames.unsqueeze(2)
        paired_frames = temporal_frames.expand(-1, -1, self.geometry.temporal_patch, -1, -1)
        return paired_frames.to(dtype)

    def validate_temporal_pixels(self, temporal_pixels, weight):
        expected = (self.geometry.channels, self.geometry.temporal_patch,
                    self.geometry.image_side, self.geometry.image_side)
        if temporal_pixels.ndim != 5 or tuple(temporal_pixels.shape[1:]) != expected:
            raise ValueError('Qwen patch projection requires complete temporal patch pairs')
        if temporal_pixels.device != weight.device:
            raise ValueError('Frames and the Qwen patch projection must be on the same device')
        if temporal_pixels.dtype != weight.dtype:
            raise ValueError('Frames and the Qwen patch projection must use the same dtype')

    def project_patches(self, temporal_pixels, weight, bias):
        self.validate_temporal_pixels(temporal_pixels, weight)
        stride = (self.geometry.temporal_patch,
                  self.geometry.patch_side, self.geometry.patch_side)
        embedded = F.conv3d(temporal_pixels, weight, bias, stride=stride)
        if embedded.shape[2] != 1:
            raise ValueError('The selected Qwen frontend projects one temporal patch per frame')
        patch_grid = embedded[:, :, 0]
        return patch_grid

    def group_patches(self, patch_grid):
        grouped_tokens = self.spatial_layout.group_projection(patch_grid)
        return grouped_tokens

    def forward(self, images, weight, bias):
        self.validate_projection(weight, bias)
        temporal_pixels = self.temporal_pair(images, weight.dtype)
        patch_grid = self.project_patches(temporal_pixels, weight, bias)
        grouped_tokens = self.group_patches(patch_grid)
        return grouped_tokens


class QwenPositionInterpolation(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.spatial_layout = QwenSpatialLayout(geometry)

    def validate_table(self, table):
        if not isinstance(table, torch.Tensor) or table.ndim != 2:
            raise ValueError('The Qwen position table must be a two-dimensional tensor')
        if table.shape[1] != self.geometry.hidden_size or not table.is_floating_point():
            raise ValueError('The Qwen position table has an incompatible embedding width or dtype')
        source_side = math.isqrt(table.shape[0])
        if source_side == 0 or source_side * source_side != table.shape[0]:
            raise ValueError('The Qwen position table must describe a square grid')
        if not bool(torch.isfinite(table).all()):
            raise ValueError('The Qwen position table contains nonfinite values')
        return source_side

    def sampling_axis(self, source_side):
        coordinates = torch.linspace(0, source_side - 1, self.geometry.patch_grid,
                                     dtype=torch.float32, device='cpu')
        lower = coordinates.int()
        upper = (lower + 1).clamp(max=source_side - 1)
        fraction = coordinates - lower
        return lower, upper, fraction

    def corner_indices(self, lower, upper, source_side):
        top_left = lower[:, None] * source_side + lower[None, :]
        top_right = lower[:, None] * source_side + upper[None, :]
        bottom_left = upper[:, None] * source_side + lower[None, :]
        bottom_right = upper[:, None] * source_side + upper[None, :]
        corners = (top_left.flatten(), top_right.flatten(),
                   bottom_left.flatten(), bottom_right.flatten())
        return torch.stack(corners, dim=0).long()

    def corner_weights(self, fraction, dtype):
        row_fraction = fraction[:, None]
        column_fraction = fraction[None, :]
        top_left = (1 - row_fraction) * (1 - column_fraction)
        top_right = (1 - row_fraction) * column_fraction
        bottom_left = row_fraction * (1 - column_fraction)
        bottom_right = row_fraction * column_fraction
        corners = (top_left.flatten(), top_right.flatten(),
                   bottom_left.flatten(), bottom_right.flatten())
        return torch.stack(corners, dim=0).to(dtype)

    def interpolate(self, table, indices, weights):
        corner_embeddings = table[indices]
        weighted_embeddings = corner_embeddings * weights[:, :, None]
        top_embeddings = weighted_embeddings[0] + weighted_embeddings[1]
        three_corners = top_embeddings + weighted_embeddings[2]
        spatial_positions = three_corners + weighted_embeddings[3]
        return spatial_positions

    def group_positions(self, spatial_positions):
        grouped_positions = self.spatial_layout.group_positions(spatial_positions)
        return grouped_positions

    def forward(self, table):
        source_side = self.validate_table(table)
        lower, upper, fraction = self.sampling_axis(source_side)
        indices = self.corner_indices(lower, upper, source_side).to(table.device)
        weights = self.corner_weights(fraction, table.dtype).to(table.device)
        spatial_positions = self.interpolate(table, indices, weights)
        grouped_positions = self.group_positions(spatial_positions)
        return grouped_positions


class VisionSpatialPooling(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry

    def validate_patch_tokens(self, positioned_tokens):
        expected = (self.geometry.patch_tokens, self.geometry.hidden_size)
        if positioned_tokens.ndim != 3 or tuple(positioned_tokens.shape[1:]) != expected:
            raise ValueError('Spatial pooling requires complete positioned Qwen patch tokens')

    def split_patch_groups(self, positioned_tokens):
        self.validate_patch_tokens(positioned_tokens)
        frames = positioned_tokens.shape[0]
        grid = self.geometry.merged_grid
        group_count = self.geometry.merge_side * self.geometry.merge_side
        width = self.geometry.hidden_size
        groups = positioned_tokens.reshape(frames, grid * grid, group_count, width)
        return groups

    def merge_patch_groups(self, positioned_tokens):
        groups = self.split_patch_groups(positioned_tokens)
        grid = self.geometry.merged_grid
        width = self.geometry.hidden_size
        merged_tokens = groups.mean(2)
        merged_grid = merged_tokens.reshape(-1, grid, grid, width)
        return merged_grid

    def split_spatial_groups(self, merged_grid):
        expected = (self.geometry.merged_grid, self.geometry.merged_grid,
                    self.geometry.hidden_size)
        if merged_grid.ndim != 4 or tuple(merged_grid.shape[1:]) != expected:
            raise ValueError('The merged Qwen grid has incompatible spatial dimensions')
        frames = merged_grid.shape[0]
        grid = self.geometry.output_grid
        group = self.geometry.pool_side
        width = self.geometry.hidden_size
        spatial_groups = merged_grid.reshape(frames, grid, group, grid, group, width)
        return spatial_groups

    def pool_spatial_grid(self, merged_grid):
        spatial_groups = self.split_spatial_groups(merged_grid)
        pooled_grid = spatial_groups.mean((2, 4))
        return pooled_grid

    def forward(self, positioned_tokens):
        merged_grid = self.merge_patch_groups(positioned_tokens)
        pooled_grid = self.pool_spatial_grid(merged_grid)
        return pooled_grid


class FrozenVisionFront(nn.Module):

    def __init__(self):
        super().__init__()
        self.geometry = VisionFrontGeometry()
        self.register_buffer('patch_weight', torch.zeros(self.geometry.kernel_shape, dtype=torch.bfloat16))
        self.register_buffer('patch_bias', torch.zeros(self.geometry.hidden_size, dtype=torch.bfloat16))
        self.register_buffer('position', torch.zeros(self.geometry.patch_tokens,
                                                   self.geometry.hidden_size, dtype=torch.bfloat16))
        self.register_buffer('ready', torch.tensor(False))
        self.patch_projection = QwenPatchProjection(self.geometry)
        self.position_interpolation = QwenPositionInterpolation(self.geometry)
        self.spatial_pooling = VisionSpatialPooling(self.geometry)

    def validate_frames(self, images):
        if not bool(self.ready):
            raise RuntimeError('Visual frontend weights have not been loaded')
        frame_batch = VisionFrameBatch.from_images(images, self.geometry)
        if images.device != self.patch_weight.device:
            raise ValueError('Video frames and the frozen frontend must be on the same device')
        return frame_batch

    @torch.no_grad()
    def prepare_position_table(self, source_table, reference_position=None):
        if not isinstance(source_table, torch.Tensor):
            raise ValueError('The source position table must be a tensor')
        if source_table.dtype != self.position.dtype:
            raise ValueError('Source and cached position tensors require the same dtype')
        source_table = source_table.detach().to('cpu').contiguous()
        prepared_position = self.position_interpolation(source_table)
        if reference_position is not None and not torch.equal(prepared_position, reference_position.detach().cpu()):
            raise ValueError('The source position table differs from the checkpoint position cache')
        self.position.copy_(prepared_position.to(self.position.device))
        return prepared_position

    def temporal_pair(self, images):
        return self.patch_projection.temporal_pair(images, self.patch_weight.dtype)

    def project_patches(self, temporal_pixels):
        return self.patch_projection.project_patches(temporal_pixels, self.patch_weight, self.patch_bias)

    def group_patches(self, patch_grid):
        return self.patch_projection.group_patches(patch_grid)

    def pool_patches(self, positioned_tokens):
        return self.spatial_pooling(positioned_tokens)

    def add_position(self, grouped_tokens):
        expected = (self.geometry.patch_tokens, self.geometry.hidden_size)
        if grouped_tokens.ndim != 3 or tuple(grouped_tokens.shape[1:]) != expected:
            raise ValueError('The grouped Qwen tokens do not match the cached positions')
        positioned_tokens = grouped_tokens + self.position
        return positioned_tokens

    def restore_token_shape(self, pooled_grid, leading_shape):
        frame_batch = VisionFrameBatch(tuple(leading_shape), self.geometry)
        return frame_batch.restore_tokens(pooled_grid)

    def forward(self, images):
        frame_batch = self.validate_frames(images)
        grouped_tokens = self.patch_projection(images, self.patch_weight, self.patch_bias)
        positioned_tokens = self.add_position(grouped_tokens)
        pooled_grid = self.spatial_pooling(positioned_tokens)
        visual_tokens = frame_batch.restore_tokens(pooled_grid)
        return visual_tokens


class QwenFrontCheckpoint:

    def __init__(self, payload, geometry):
        self.payload = payload
        self.geometry = geometry
        self.prefix = 'vision_frontend.'

    def model_state(self):
        state = self.payload.get('model')
        if not isinstance(state, dict):
            raise ValueError('The inference checkpoint requires a model state dictionary')
        return state

    def tensor(self, state, name, shape, dtype):
        key = self.prefix + name
        value = state.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError('The frozen Qwen frontend is missing ' + key)
        if tuple(value.shape) != shape:
            raise ValueError('The frozen Qwen frontend has an incompatible shape for ' + key)
        if value.dtype != dtype:
            raise ValueError('The frozen Qwen frontend has an incompatible dtype for ' + key)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('The frozen Qwen frontend contains nonfinite values in ' + key)
        return value

    def validate_projection(self, state):
        weight = self.tensor(state, 'patch_weight', self.geometry.kernel_shape, torch.bfloat16)
        bias = self.tensor(state, 'patch_bias', (self.geometry.hidden_size,), torch.bfloat16)
        ready = self.tensor(state, 'ready', (), torch.bool)
        return weight, bias, ready

    def source_position_table(self):
        frontend = self.payload.get('frontend')
        if frontend is None:
            if self.payload['schema'] == 3:
                raise ValueError('The frontend checkpoint requires its original Qwen position table')
            return None
        if not isinstance(frontend, dict) or set(frontend) != {'position_table'}:
            raise ValueError('Expected the original Qwen frontend position table')
        table = frontend['position_table']
        if not isinstance(table, torch.Tensor):
            raise ValueError('The original Qwen position table must be a tensor')
        if table.dtype != torch.bfloat16:
            raise ValueError('The original Qwen position table must retain its bfloat16 dtype')
        return table

    def cached_position(self, state, required):
        key = self.prefix + 'position'
        if key not in state and not required:
            return None
        shape = (self.geometry.patch_tokens, self.geometry.hidden_size)
        return self.tensor(state, 'position', shape, torch.bfloat16)

    def restore_state(self, frontend):
        if frontend.geometry != self.geometry:
            raise ValueError('Checkpoint and Qwen frontend geometries differ')
        state = self.model_state()
        self.validate_projection(state)
        source_table = self.source_position_table()
        reference = self.cached_position(state, required=source_table is None)
        if source_table is None:
            return state
        prepared_position = frontend.prepare_position_table(source_table, reference_position=reference)
        restored_state = dict(state)
        restored_state[self.prefix + 'position'] = prepared_position
        return restored_state


class VisionAdapter(nn.Sequential):

    def __init__(self, config):
        normalization = nn.LayerNorm(1024)
        projection = nn.Linear(
            in_features=1024,
            out_features=config.width,
        )
        super().__init__(normalization, projection)

    def forward(self, visual_embeddings):
        normalization, projection = self
        normalized_embeddings = normalization(visual_embeddings)
        visual_tokens = projection(normalized_embeddings)
        return visual_tokens


class LanguageAdapter(nn.Sequential):

    def __init__(self, config):
        normalization = nn.LayerNorm(2048)
        projection = nn.Linear(
            in_features=2048,
            out_features=config.width,
        )
        super().__init__(normalization, projection)

    def forward(self, prompt_embeddings):
        normalization, projection = self
        normalized_embeddings = normalization(prompt_embeddings)
        language_tokens = projection(normalized_embeddings)
        return language_tokens


class VisionMerger(nn.Sequential):

    def __init__(self, config):
        summary_width = 2 * config.width
        normalization = nn.LayerNorm(summary_width)
        projection = nn.Linear(
            in_features=summary_width,
            out_features=config.width,
        )
        activation = nn.ReLU()
        super().__init__(normalization, projection, activation)

    def forward(self, pooled_summary):
        normalization, projection, activation = self
        normalized_summary = normalization(pooled_summary)
        projected_summary = projection(normalized_summary)
        visual_summary = activation(projected_summary)
        return visual_summary


class QualityReadout(nn.Sequential):

    def __init__(self, config):
        sequence_length = config.frames + config.prompt_length
        input_features = sequence_length * 96
        flatten = nn.Flatten(start_dim=-2)
        hidden_projection = nn.Linear(
            in_features=input_features,
            out_features=config.readout_hidden,
        )
        activation = nn.ReLU()
        grade_projection = nn.Linear(
            in_features=config.readout_hidden,
            out_features=len(QUALITY_WORDS),
        )
        super().__init__(flatten, hidden_projection, activation, grade_projection)

    def forward(self, sequence):
        flatten, hidden_projection, activation, grade_projection = self
        flattened_sequence = flatten(sequence)
        hidden_features = hidden_projection(flattened_sequence)
        activated_features = activation(hidden_features)
        grade_logits = grade_projection(activated_features)
        return grade_logits


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
        origins = []
        for row in range(4):
            top = 23 + 119 * row
            for column in range(4):
                left = 23 + 119 * column
                origins.append((top, left))
        return origins

    @property
    def frame_offsets(self):
        return ((0, 0), (0, 59), (59, 0), (59, 59))

    def fields(self, tokens):
        if self.is_vision:
            amplitude = self.encode_vision_field(tokens)
        else:
            amplitude = self.encode_language_field(tokens)
        intensity = amplitude.square()
        mean_intensity = intensity.mean((-2, -1), keepdim=True)
        scale = mean_intensity.clamp_min(1e-12).sqrt()
        normalized_amplitude = amplitude / scale
        return normalized_amplitude

    def encode_vision_field(self, visual_tokens):
        channel_projection = self.width_to_field(visual_tokens)
        channel_amplitude = F.relu(channel_projection)
        transposed_amplitude = channel_amplitude.transpose(-2, -1)
        spatial_projection = self.tokens_to_field(transposed_amplitude)
        spatial_amplitude = F.relu(spatial_projection)
        frame_amplitude = spatial_amplitude.transpose(-2, -1)
        return frame_amplitude

    def encode_language_field(self, joint_tokens):
        channel_projection = self.width_to_field(joint_tokens)
        token_amplitude = F.relu(channel_projection)
        missing_rows = 56 - token_amplitude.shape[-2]
        video_amplitude = F.pad(token_amplitude, (0, 0, 0, missing_rows))
        return video_amplitude

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
        if self.is_vision:
            return self.place_vision_fields(fields, canvas)
        return self.place_language_fields(fields, canvas)

    def place_vision_fields(self, fields, canvas):
        for slot, (row, column) in enumerate(self.origins):
            for frame, (offset_y, offset_x) in enumerate(self.frame_offsets):
                top = row + offset_y + 14
                left = column + offset_x + 14
                bottom = top + 27
                right = left + 27
                frame_amplitude = fields[:, slot, frame]
                canvas[:, top:bottom, left:right] = frame_amplitude
        return canvas

    def place_language_fields(self, fields, canvas):
        for slot, (row, column) in enumerate(self.origins):
            top = row + 29
            left = column + 29
            bottom = top + 56
            right = left + 56
            video_amplitude = fields[:, slot]
            canvas[:, top:bottom, left:right] = video_amplitude
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
        if self.is_vision:
            patches = self.vision_detector_regions(detector)
        else:
            patches = self.language_detector_regions(detector)
        nonnegative_patches = patches.clamp_min(0)
        relative_intensity = self.process_detector(nonnegative_patches)
        return relative_intensity

    def vision_detector_regions(self, detector):
        video_regions = []
        for row, column in self.origins:
            frame_regions = []
            for offset_y, offset_x in self.frame_offsets:
                top = row + offset_y
                left = column + offset_x
                bottom = top + 56
                right = left + 56
                frame_region = detector[:, top:bottom, left:right]
                frame_regions.append(frame_region)
            video_region = torch.stack(frame_regions, dim=1)
            video_regions.append(video_region)
        return torch.stack(video_regions, dim=1)

    def language_detector_regions(self, detector):
        video_regions = []
        for row, column in self.origins:
            bottom = row + 115
            right = column + 115
            video_region = detector[:, row:bottom, column:right]
            video_regions.append(video_region)
        return torch.stack(video_regions, dim=1)

    def detector_tokens(self, patches, count):
        leading = patches.shape[:-2]
        detector_images = patches.reshape(-1, 1, *patches.shape[-2:])
        pooled_images = F.adaptive_avg_pool2d(detector_images, (count, 96))
        pooled_tokens = pooled_images[:, 0]
        normalized_tokens = self.norm(pooled_tokens)
        activated_tokens = F.relu(normalized_tokens)
        projected_tokens = self.output(activated_tokens)
        output_shape = (*leading, count, projected_tokens.shape[-1])
        detector_tokens = projected_tokens.reshape(*output_shape)
        return detector_tokens

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
        self.vision_adapter = VisionAdapter(config)
        self.language_adapter = LanguageAdapter(config)
        self.vision_router = Router(config, 49)
        self.vision_experts = Experts(config)
        self.vision_global = Global(config, 'vision')
        self.vision_merger = VisionMerger(config)
        self.vision_frame_position = nn.Parameter(torch.empty(1, 1, 4, config.width))
        self.language_sequence_position = nn.Parameter(torch.empty(1, 64, config.width))
        self.language_router = Router(config, 4 + config.prompt_length)
        self.language_experts = Experts(config)
        self.language_global = Global(config, 'language')
        self.readout = QualityReadout(config)
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
        projected_tokens = self.vision_adapter(visual_embeddings.float())
        routing = self.vision_router(projected_tokens)
        expert_tokens = self.vision_experts(projected_tokens, routing['weights'])
        detector_tokens = self.vision_global(expert_tokens)
        average_tokens = detector_tokens.mean(-2)
        maximum_tokens = detector_tokens.amax(-2)
        pooled_summary = torch.cat((average_tokens, maximum_tokens), dim=-1)
        visual_summary = self.vision_merger(pooled_summary)
        positioned_summary = visual_summary + self.vision_frame_position
        return positioned_summary, routing

    def encode_language(self, visual_summary):
        projected_prompt = self.language_adapter(self.language_prompt_embedding.float())
        prompt_tokens = projected_prompt[None, None].expand(len(visual_summary), 16, -1, -1)
        joint_tokens = torch.cat((visual_summary, prompt_tokens), dim=-2)
        sequence_position = self.language_sequence_position[:, None, :joint_tokens.shape[-2]]
        positioned_tokens = joint_tokens + sequence_position
        routing = self.language_router(positioned_tokens)
        expert_tokens = self.language_experts(positioned_tokens, routing['weights'])
        detector_tokens = self.language_global(expert_tokens)
        return detector_tokens, routing

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
    if payload['schema'] not in (2, 3) or payload['architecture'] not in (ARCHITECTURE, LEGACY_ARCHITECTURE):
        raise ValueError('Expected the LightGen-II inference checkpoint')
    config = dict(payload['config'])
    config.setdefault('shared_global_phase', False)
    if config['shared_global_phase'] != (payload['architecture'] == ARCHITECTURE):
        raise ValueError('Global phase sharing conflicts with the checkpoint architecture')
    model = LightGen2(ModelConfig(**config))
    frontend_checkpoint = QwenFrontCheckpoint(payload, model.vision_frontend.geometry)
    state = frontend_checkpoint.restore_state(model.vision_frontend)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload['metadata']
