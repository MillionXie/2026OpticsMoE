import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputTopKRouter(nn.Module):
    """Standard input-dependent sparse MoE gate: pooled image -> Linear -> top-k."""

    def __init__(self, num_experts=9, top_k=3, pool_size=10, temperature=1.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("prompt.top_k must be between 1 and 9.")
        if self.temperature <= 0:
            raise ValueError("prompt.temperature must be positive.")
        self.gate = nn.Linear(self.pool_size * self.pool_size, self.num_experts)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, images):
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.shape[1] != 1:
            images = images.mean(1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(images.float(), (self.pool_size, self.pool_size)).flatten(1)
        logits = self.gate(pooled)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        _, indices = torch.topk(probabilities, k=self.top_k, dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected.scatter_(1, indices, True)
        sparse = probabilities * selected.to(probabilities.dtype)
        weights = sparse / (sparse.sum(-1, keepdim=True) + 1e-8)
        importance = probabilities.mean(0)
        load = selected.float().mean(0) / float(self.top_k)
        balance_loss = float(self.num_experts) * torch.sum(importance * load)
        # Differentiable importance regularization.  Because importance sums
        # to one, n*sum(importance**2)-1 is zero for uniform routing and grows
        # to n-1 when all probability mass collapses to one expert.  Unlike
        # the hard top-k load, every softmax logit receives a gradient here.
        importance_loss = (
            float(self.num_experts) * torch.sum(importance.square()) - 1.0
        )
        normalized_entropy = (
            -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1).mean()
            / math.log(float(self.num_experts))
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "balance_loss": balance_loss,
            "importance_loss": importance_loss,
            "normalized_entropy": normalized_entropy,
            "importance": importance,
            "load": load,
        }


class GlobalRouterPrompt(nn.Module):
    """Region-amplitude prompt with one continuous global lens phase.

    The 3x3 partition exists only in the amplitude plane.  The phase is built
    once on a single global coordinate system.  Consequently it has no cell
    resets or stitched local lenses.  When the global quadratic phase is
    expressed around any cell centre it decomposes into a local quadratic lens
    term plus a linear carrier (grating) term, which is why this one continuous
    phase implements both focusing and the nine spatial carrier directions.
    """

    def __init__(
        self,
        layout,
        wavelength_m,
        pixel_size_m,
        input_to_prompt_m,
        propagation_m,
        focal_length_m,
        top_k=3,
        pool_size=10,
        temperature=1.0,
        grating_sign_x=1.0,
        grating_sign_y=1.0,
        min_grating_period_pixels=0.0,
        mode="region_amplitude_global_lens",
    ):
        super().__init__()
        layout.validate()
        if mode != "region_amplitude_global_lens":
            raise ValueError("prompt.mode must be 'region_amplitude_global_lens'.")
        if float(focal_length_m) <= 0:
            raise ValueError("optics.prompt_focal_length_m must be positive.")
        self.layout = layout
        self.mode = mode
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.focal_length_m = float(focal_length_m)
        self.convolution_distance_m = float(propagation_m)
        self.router_network = InputTopKRouter(9, top_k, pool_size, temperature)

        cell_masks = []
        for row in range(3):
            for col in range(3):
                y0 = layout.active_start + row * layout.expert_pitch
                x0 = layout.active_start + col * layout.expert_pitch
                mask = torch.zeros((layout.canvas_size, layout.canvas_size), dtype=torch.float32)
                mask[y0 : y0 + layout.expert_pitch, x0 : x0 + layout.expert_pitch] = 1.0
                cell_masks.append(mask)

        center = layout.canvas_size // 2
        axis = (torch.arange(layout.canvas_size, dtype=torch.float64) - center) * self.pixel_size_m
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        # One global coordinate system, one global phase.  Do not move the
        # origin to each amplitude cell: doing so creates the rejected 3x3
        # stitched phase plate.
        global_lens_phase = -math.pi / (self.wavelength_m * self.focal_length_m) * (
            x_grid.square() + y_grid.square()
        )
        active_mask = layout.active_mask().float()
        wrapped_phase = torch.remainder(global_lens_phase, 2.0 * math.pi).float() * active_mask

        # Diagnostic only: the largest local slope of the global quadratic
        # phase, expressed as an equivalent linear grating period.
        max_radius_m = 0.5 * layout.active_size * self.pixel_size_m
        max_frequency = max_radius_m / (self.wavelength_m * self.focal_length_m)
        self.max_abs_grating_frequency = max_frequency
        self.nyquist_frequency = 1.0 / (2.0 * self.pixel_size_m)
        self.min_grating_period_pixels = float(min_grating_period_pixels)
        self.edge_grating_period_pixels = (
            math.inf if max_frequency == 0 else 1.0 / (max_frequency * self.pixel_size_m)
        )

        self.register_buffer("cell_masks", torch.stack(cell_masks), persistent=False)
        self.register_buffer("active_mask", active_mask, persistent=False)
        self.register_buffer("global_lens_phase", wrapped_phase, persistent=False)
        # Kept as a fixed buffer for checkpoint/report compatibility.  It is
        # deliberately not applied per cell and is never trainable.
        self.register_buffer("phase_biases", torch.zeros(9), persistent=True)

    def amplitude_map(self, weights):
        """Physical prompt amplitude: one uniform routing weight per 150x150 cell."""
        return torch.einsum("be,ehw->bhw", weights, self.cell_masks.to(device=weights.device, dtype=weights.dtype))

    def phase_map(self):
        """One continuous wrapped global lens/carrier phase over active 450x450."""
        return self.global_lens_phase

    def transmission(self, weights):
        amplitude = self.amplitude_map(weights)
        phase = self.phase_map().to(device=weights.device)
        return amplitude.to(torch.complex64) * torch.exp(1j * phase).to(torch.complex64).unsqueeze(0)

    def routing(self, images):
        routing = self.router_network(images)
        routing["transmission"] = self.transmission(routing["weights"])
        routing["prompt_amplitude"] = self.amplitude_map(routing["weights"])
        routing["prompt_phase"] = self.phase_map().unsqueeze(0).expand(images.shape[0], -1, -1)
        return routing

    def forward(self, field, images):
        # Compatibility path.  The classifier uses routing() and the global
        # convolution operator instead of treating this as a pointwise ASM
        # prompt plane.
        routing = self.routing(images)
        return field.to(torch.complex64) * routing["transmission"], routing
