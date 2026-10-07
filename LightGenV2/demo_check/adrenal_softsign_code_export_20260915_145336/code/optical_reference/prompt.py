import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .optics import AngularSpectrumPropagator, PhaseLayer


def _topk_routing(probabilities, top_k):
    """Apply hard Top-k selection while keeping selected soft weights differentiable."""
    num_experts = probabilities.shape[-1]
    if int(top_k) == num_experts:
        # Dense optical routing: retain all continuously weighted paths.
        # No top-k operator and no sample-dependent pruning in this branch.
        selected = torch.ones_like(probabilities, dtype=torch.bool)
        indices = torch.arange(num_experts,device=probabilities.device).expand(probabilities.shape[0],-1)
    else:
        _, indices = torch.topk(probabilities, k=int(top_k), dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected.scatter_(1, indices, True)
    sparse = probabilities * selected.to(probabilities.dtype)
    # Preserve the normalization used by the original electronic router so
    # that adding the optical ablation does not silently alter that baseline.
    weights = sparse / (sparse.sum(-1, keepdim=True) + 1.0e-8)
    importance = probabilities.mean(0)
    load = selected.float().mean(0) / float(top_k)
    balance_loss = float(num_experts) * torch.sum(importance * load)
    importance_loss = float(num_experts) * torch.sum(importance.square()) - 1.0
    normalized_entropy = (
        -(probabilities.clamp_min(1.0e-12).log() * probabilities).sum(-1).mean()
        / math.log(float(num_experts))
    )
    return {
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


class InputTopKRouter(nn.Module):
    """Standard input-dependent sparse MoE gate: pooled image -> Linear -> top-k."""

    def __init__(self, num_experts=4, top_k=2, pool_size=10, temperature=1.0):
        super().__init__()
        self.is_optical_router = False
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(f"prompt.top_k must be between 1 and {self.num_experts}.")
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
        result = _topk_routing(probabilities, self.top_k)
        result.update({
            "logits": logits,
            "scores": logits,
            "temperature": logits.new_tensor(self.temperature),
            "router_type": "input_topk",
        })
        return result


def make_rectangular_region_masks(height, width, boxes):
    """Build validated, non-overlapping CCD integration masks from config boxes."""
    height = int(height)
    width = int(width)
    masks = torch.zeros(len(boxes), height, width, dtype=torch.float32)
    for index, bounds in enumerate(boxes):
        if len(bounds) != 4:
            raise ValueError("Each optical-router detector region must be [y0,y1,x0,x1]")
        y0, y1, x0, x1 = (int(value) for value in bounds)
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"Invalid optical-router detector region {index}: {list(bounds)}")
        masks[index, y0:y1, x0:x1] = 1.0
    if masks.numel() and torch.any(masks.sum(0) > 1.0):
        raise ValueError("Optical-router detector regions must not overlap")
    return masks


class RegionIntegrator(nn.Module):
    """CCD regional energy integration without constructing dense selection matrices."""

    def __init__(self, region_masks, reduce="sum"):
        super().__init__()
        if region_masks.ndim != 3:
            raise ValueError("region_masks must have shape [E,H,W]")
        if reduce not in {"sum", "mean"}:
            raise ValueError("optical_router.detector_reduce must be 'sum' or 'mean'")
        self.reduce = str(reduce)
        self.register_buffer("region_masks", region_masks.float(), persistent=False)
        self.register_buffer(
            "region_areas", region_masks.float().sum((-2, -1)).clamp_min(1.0), persistent=False
        )

    def forward(self, intensity):
        scores = torch.einsum("bhw,ehw->be", intensity, self.region_masks)
        if self.reduce == "mean":
            scores = scores / self.region_areas.unsqueeze(0)
        return scores


class OpticalTopKRouter(nn.Module):
    """Trainable phase SLM -> ASM -> CCD regions -> softmax -> Top-k router."""

    def __init__(
        self,
        num_experts,
        top_k,
        grid_size,
        detector_regions,
        wavelength_m,
        pixel_size_m,
        propagation_distance_m,
        temperature=1.0,
        learnable_temperature=False,
        detector_reduce="sum",
        evanescent_mode="zero",
        phase_parameterization="sigmoid",
        phase_init="zeros",
        phase_init_std=0.02,
        phase_trainable=True,
        k_space_constraint_enabled=False,
        theta_max_deg=1.0,
    ):
        super().__init__()
        self.is_optical_router = True
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        if isinstance(grid_size, int):
            height = width = int(grid_size)
        else:
            height, width = (int(value) for value in grid_size)
        self.grid_size = (height, width)
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(f"prompt.top_k must be between 1 and {self.num_experts}.")
        if len(detector_regions) != self.num_experts:
            raise ValueError(
                f"optical_router.detector_regions must contain {self.num_experts} boxes"
            )
        if float(temperature) <= 0.0:
            raise ValueError("optical_router.temperature must be positive")
        self.slm = PhaseLayer(
            self.grid_size,
            parameterization=str(phase_parameterization),
            init=str(phase_init),
            init_std=float(phase_init_std),
            phase_dropout_mode="none",
            phase_dropout_p=0.0,
        )
        self.slm.raw_phase.requires_grad_(bool(phase_trainable))
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=float(wavelength_m),
            pixel_size_m=float(pixel_size_m),
            grid_size=self.grid_size,
            distance_m=float(propagation_distance_m),
            evanescent_mode=str(evanescent_mode),
            k_space_constraint_enabled=bool(k_space_constraint_enabled),
            theta_max_deg=float(theta_max_deg),
        )
        masks = make_rectangular_region_masks(height, width, detector_regions)
        self.detector = RegionIntegrator(masks, reduce=detector_reduce)
        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            initial = float(temperature)
            inverse_softplus = initial if initial > 20.0 else math.log(math.expm1(initial))
            self.raw_temperature = nn.Parameter(torch.tensor(inverse_softplus, dtype=torch.float32))
            self.register_buffer("fixed_temperature", torch.tensor(0.0), persistent=False)
        else:
            self.raw_temperature = None
            self.register_buffer(
                "fixed_temperature", torch.tensor(float(temperature), dtype=torch.float32)
            )

    def get_temperature(self):
        if self.raw_temperature is None:
            return self.fixed_temperature
        return F.softplus(self.raw_temperature) + 1.0e-6

    def encode_input(self, images):
        if images.ndim == 4:
            if images.shape[1] != 1:
                images = images.mean(1, keepdim=True)
            images = images[:, 0]
        if images.ndim != 3:
            raise ValueError("Optical router input must have shape [B,H,W] or [B,1,H,W]")
        if tuple(images.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"Optical router expected input grid {self.grid_size}, got {tuple(images.shape[-2:])}"
            )
        return images.float().clamp_min(0.0).to(torch.complex64)

    def forward(self, images):
        encoded = self.encode_input(images)
        field = self.propagator(self.slm(encoded))
        intensity = field.abs().square()
        scores = self.detector(intensity)
        temperature = self.get_temperature().to(device=scores.device, dtype=scores.dtype)
        probabilities = torch.softmax(scores / temperature, dim=-1)
        result = _topk_routing(probabilities, self.top_k)
        result.update(
            {
                "logits": scores,
                "scores": scores,
                "temperature": temperature,
                "router_type": "optical_topk",
                "field": field,
                "intensity": intensity,
                "phase": self.slm.get_phase_wrapped(),
                "score_mean": scores.mean(),
                "score_std": scores.std(unbiased=False),
            }
        )
        return result


class GlobalRouterPrompt(nn.Module):
    """Region-amplitude prompt with one continuous global lens phase.

    The 2x2 expert partition exists only in the amplitude plane.  The phase is built
    once on a single global coordinate system.  Consequently it has no cell
    resets or stitched local lenses.  When the global quadratic phase is
    expressed around any cell centre it decomposes into a local quadratic lens
    term plus a linear carrier (grating) term, which is why this one continuous
    phase implements both focusing and the four spatial carrier directions.
    """

    def __init__(
        self,
        layout,
        wavelength_m,
        pixel_size_m,
        input_to_prompt_m,
        propagation_m,
        focal_length_m,
        top_k=2,
        pool_size=10,
        temperature=1.0,
        grating_sign_x=1.0,
        grating_sign_y=1.0,
        min_grating_period_pixels=0.0,
        mode="region_amplitude_global_lens",
        routing_type="input_topk",
        optical_router_cfg=None,
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
        self.routing_type = str(routing_type).lower()
        if self.routing_type == "input_topk":
            self.router_network = InputTopKRouter(layout.num_experts, top_k, pool_size, temperature)
        elif self.routing_type == "optical_topk":
            router_cfg = dict(optical_router_cfg or {})
            grid_size = router_cfg.get("grid_size", layout.input_size)
            self.router_network = OpticalTopKRouter(
                num_experts=layout.num_experts,
                top_k=top_k,
                grid_size=grid_size,
                detector_regions=router_cfg.get("detector_regions", []),
                wavelength_m=float(router_cfg.get("wavelength_m", wavelength_m)),
                pixel_size_m=float(router_cfg.get("pixel_size_m", pixel_size_m)),
                propagation_distance_m=float(router_cfg.get("propagation_distance_m", 0.05)),
                temperature=float(router_cfg.get("temperature", temperature)),
                learnable_temperature=bool(router_cfg.get("learnable_temperature", False)),
                detector_reduce=str(router_cfg.get("detector_reduce", "sum")),
                evanescent_mode=str(router_cfg.get("evanescent_mode", "zero")),
                phase_parameterization=str(router_cfg.get("phase_parameterization", "sigmoid")),
                phase_init=str(router_cfg.get("phase_init", "zeros")),
                phase_init_std=float(router_cfg.get("phase_init_std", 0.02)),
                phase_trainable=bool(router_cfg.get("phase_trainable", True)),
                k_space_constraint_enabled=bool(router_cfg.get("k_space_constraint_enabled", False)),
                theta_max_deg=float(router_cfg.get("theta_max_deg", 1.0)),
            )
        else:
            raise ValueError("prompt.routing_type must be 'input_topk' or 'optical_topk'")

        # The routing amplitude is applied only on the four physical expert
        # apertures.  Thus every 224x224 routing region is aligned pixel for
        # pixel with its corresponding local expert mask; the 30-pixel cross
        # gap remains dark.
        cell_masks = layout.expert_masks().float()

        center = layout.canvas_size // 2
        axis = (torch.arange(layout.canvas_size, dtype=torch.float64) - center) * self.pixel_size_m
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        # One global coordinate system, one global phase.  Do not move the
        # origin to each amplitude cell: doing so creates a stitched phase
        # plate.
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

        self.register_buffer("cell_masks", cell_masks, persistent=False)
        self.register_buffer("active_mask", active_mask, persistent=False)
        self.register_buffer("global_lens_phase", wrapped_phase, persistent=False)
        # Kept as a fixed buffer for checkpoint/report compatibility.  It is
        # deliberately not applied per cell and is never trainable.
        self.register_buffer("phase_biases", torch.zeros(layout.num_experts), persistent=True)

    def amplitude_map(self, weights):
        """Physical prompt amplitude: one uniform routing weight per 224x224 expert aperture."""
        return torch.einsum("be,ehw->bhw", weights, self.cell_masks.to(device=weights.device, dtype=weights.dtype))

    def phase_map(self):
        """One continuous wrapped global lens/carrier phase over active 478x478."""
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
