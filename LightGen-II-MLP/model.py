from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'lightgen2_temporal_mlp_global_v1'
QUALITY_WORDS = ('Bad', 'Poor', 'Fair', 'Good', 'Excellent')


@dataclass(frozen=True)
class ModelConfig:
    width: int = 192
    global_hidden: int = 32
    router_hidden: int = 128
    readout_hidden: int = 32
    prompt_length: int = 38
    videos: int = 16
    frames: int = 4
    tokens: int = 49
    temperature: float = 1.1

    def __post_init__(self):
        dimensions=(self.width,self.router_hidden,self.readout_hidden,self.prompt_length,
                    self.videos,self.frames,self.tokens)
        if dimensions!=(192,128,32,38,16,4,49):
            raise ValueError('Checkpoint dimensions differ from the MLP reference architecture')
        if type(self.global_hidden) is not int or self.global_hidden<=0:
            raise ValueError('The global MLP hidden width must be a positive integer')
        if not math.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError('The router temperature must be finite and positive')



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
        first_coordinate = 0
        last_coordinate = source_side - 1
        target_positions = self.geometry.patch_grid
        coordinates = torch.linspace(
            start=first_coordinate,
            end=last_coordinate,
            steps=target_positions,
            dtype=torch.float32,
            device='cpu',
        )
        lower = coordinates.int()
        next_coordinate = lower + 1
        upper = next_coordinate.clamp(max=last_coordinate)
        fraction = coordinates - lower
        return lower, upper, fraction

    def corner_indices(self, lower, upper, source_side):
        top_rows = lower[:, None]
        bottom_rows = upper[:, None]
        left_columns = lower[None, :]
        right_columns = upper[None, :]
        top_row_offsets = top_rows * source_side
        bottom_row_offsets = bottom_rows * source_side
        top_left = top_row_offsets + left_columns
        top_right = top_row_offsets + right_columns
        bottom_left = bottom_row_offsets + left_columns
        bottom_right = bottom_row_offsets + right_columns
        top_left_indices = top_left.flatten()
        top_right_indices = top_right.flatten()
        bottom_left_indices = bottom_left.flatten()
        bottom_right_indices = bottom_right.flatten()
        corner_indices = torch.stack((top_left_indices, top_right_indices,
                                      bottom_left_indices, bottom_right_indices), dim=0)
        return corner_indices.long()

    def corner_weights(self, fraction, dtype):
        row_fraction = fraction[:, None]
        column_fraction = fraction[None, :]
        top_row_weight = 1 - row_fraction
        bottom_row_weight = row_fraction
        left_column_weight = 1 - column_fraction
        right_column_weight = column_fraction
        top_left = top_row_weight * left_column_weight
        top_right = top_row_weight * right_column_weight
        bottom_left = bottom_row_weight * left_column_weight
        bottom_right = bottom_row_weight * right_column_weight
        top_left_weights = top_left.flatten()
        top_right_weights = top_right.flatten()
        bottom_left_weights = bottom_left.flatten()
        bottom_right_weights = bottom_right.flatten()
        corner_weights = torch.stack((top_left_weights, top_right_weights,
                                      bottom_left_weights, bottom_right_weights), dim=0)
        return corner_weights.to(dtype)

    def interpolate(self, table, indices, weights):
        top_left_indices = indices[0]
        top_right_indices = indices[1]
        bottom_left_indices = indices[2]
        bottom_right_indices = indices[3]
        top_left_embeddings = table[top_left_indices]
        top_right_embeddings = table[top_right_indices]
        bottom_left_embeddings = table[bottom_left_indices]
        bottom_right_embeddings = table[bottom_right_indices]
        top_left_weights = weights[0, :, None]
        top_right_weights = weights[1, :, None]
        bottom_left_weights = weights[2, :, None]
        bottom_right_weights = weights[3, :, None]
        weighted_top_left = top_left_embeddings * top_left_weights
        weighted_top_right = top_right_embeddings * top_right_weights
        weighted_bottom_left = bottom_left_embeddings * bottom_left_weights
        weighted_bottom_right = bottom_right_embeddings * bottom_right_weights
        top_embeddings = weighted_top_left + weighted_top_right
        three_corners = top_embeddings + weighted_bottom_left
        spatial_positions = three_corners + weighted_bottom_right
        return spatial_positions

    def group_positions(self, spatial_positions):
        grouped_positions = self.spatial_layout.group_positions(spatial_positions)
        return grouped_positions

    def forward(self, table):
        source_side = self.validate_table(table)
        lower, upper, fraction = self.sampling_axis(source_side)
        sampling_indices = self.corner_indices(lower, upper, source_side)
        sampling_weights = self.corner_weights(fraction, table.dtype)
        indices = sampling_indices.to(table.device)
        weights = sampling_weights.to(table.device)
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
        source_table = source_table.detach()
        source_table = source_table.to('cpu')
        source_table = source_table.contiguous()
        prepared_position = self.position_interpolation(source_table)
        if reference_position is not None:
            if not isinstance(reference_position, torch.Tensor):
                raise ValueError('The checkpoint position cache must be a tensor')
            if reference_position.shape != prepared_position.shape:
                raise ValueError('The checkpoint position cache has incompatible dimensions')
            if reference_position.dtype != prepared_position.dtype:
                raise ValueError('The checkpoint position cache has an incompatible dtype')
            reference_position = reference_position.detach()
            reference_position = reference_position.to('cpu')
            if not torch.equal(prepared_position, reference_position):
                raise ValueError('The source position table differs from the checkpoint position cache')
        destination_position = prepared_position.to(self.position.device)
        self.position.copy_(destination_position)
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


class GlobalMLP(nn.Module):

    def __init__(self, config, token_count, output_width):
        super().__init__()
        self.token_count = token_count
        self.input_width = config.width
        self.output_width = output_width
        self.fc1 = nn.Linear(token_count * config.width, config.global_hidden)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(config.global_hidden, token_count * output_width)

    def forward(self, tokens):
        if tuple(tokens.shape[-2:]) != (self.token_count, self.input_width):
            raise ValueError('Global MLP input dimensions differ')
        leading = tokens.shape[:-2]
        flattened = tokens.flatten(-2)
        hidden = self.activation(self.fc1(flattened))
        output = self.fc2(hidden)
        return output.reshape(*leading, self.token_count, self.output_width)



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
        self.vision_global = GlobalMLP(config, config.tokens, config.width)
        self.vision_merger = VisionMerger(config)
        self.vision_frame_position = nn.Parameter(torch.empty(1, 1, 4, config.width))
        self.language_sequence_position = nn.Parameter(torch.empty(1, 64, config.width))
        self.language_router = Router(config, 4 + config.prompt_length)
        self.language_experts = Experts(config)
        self.language_global = GlobalMLP(config, config.frames + config.prompt_length, 96)
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
        global_tokens = self.vision_global(expert_tokens)
        average_tokens = global_tokens.mean(-2)
        maximum_tokens = global_tokens.amax(-2)
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
        global_tokens = self.language_global(expert_tokens)
        return global_tokens, routing

    def forward(self, normalized_rgb):
        visual_embeddings = self.vision_frontend(normalized_rgb)
        return self.predict_from_embeddings(visual_embeddings)

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
    if payload['schema'] != 3 or payload['architecture'] != ARCHITECTURE:
        raise ValueError('Expected a LightGen-II MLP global checkpoint')
    model = LightGen2(ModelConfig(**payload['config']))
    frontend = QwenFrontCheckpoint(payload, model.vision_frontend.geometry)
    model.load_state_dict(frontend.restore_state(model.vision_frontend), strict=True)
    return model.to(device).eval(), payload['metadata']

