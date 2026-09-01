from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    AngularSpectrumPropagator,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.router import (
    ElectronicAmplitudeRouter,
)


ROUTING_NORMALIZATIONS = {"legacy_l1", "power_l2"}


def sparsify_probabilities(
    probabilities: torch.Tensor,
    top_k: int,
    *,
    normalization: str,
    straight_through: bool,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create hard top-k weights with an optional dense surrogate gradient.

    ``legacy_l1`` exactly reproduces the original router: selected amplitudes
    sum to one. ``power_l2`` makes the selected *amplitudes* have unit L2 norm,
    so changing k does not silently change total routed optical power.

    The forward value is always genuinely sparse.  With straight-through
    enabled its gradient is that of the corresponding dense normalization;
    this is essential for k=1, whose legacy selected weight is identically 1.
    """

    if normalization not in ROUTING_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {sorted(ROUTING_NORMALIZATIONS)}"
        )
    top_k = int(top_k)
    if not 1 <= top_k <= probabilities.shape[-1]:
        raise ValueError("top_k must be between one and the expert count")
    _, indices = torch.topk(probabilities, top_k, dim=-1)
    selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(
        1, indices, True
    )
    sparse = probabilities * selected
    if normalization == "legacy_l1":
        hard = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        dense = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
    else:
        hard = sparse / sparse.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(eps)
        dense = probabilities / probabilities.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(eps)
    # Standard straight-through estimator: the numerical forward is exactly
    # ``hard`` while the backward Jacobian is exactly that of ``dense``.
    # Writing ``hard + dense - dense.detach()`` would incorrectly retain the
    # hard-branch gradient as well (and doubles the k=4 dense gradient).
    weights = hard.detach() + dense - dense.detach() if straight_through else hard
    return weights, selected, indices


class FairElectronicAmplitudeRouter(ElectronicAmplitudeRouter):
    """Original 788-parameter electronic router with an explicit power contract."""

    def __init__(self, geometry: Any, settings: Any) -> None:
        super().__init__(
            geometry,
            settings.top_k,
            settings.router_pool_size,
            settings.router_temperature,
            settings.router_input_layernorm_enabled,
            settings.router_input_layernorm_eps,
            noise_std=settings.router_noise_std,
            gate_init_std=getattr(settings, "router_gate_init_std", 0.01),
        )
        self.weight_normalization = str(settings.router_weight_normalization)
        self.straight_through = bool(settings.router_straight_through)

    def forward(self, input_fields: torch.Tensor) -> dict[str, torch.Tensor | str | bool]:
        routing = super().forward(input_fields)
        weights, selected, indices = sparsify_probabilities(
            routing["probabilities"],
            self.router.top_k,
            normalization=self.weight_normalization,
            straight_through=self.straight_through,
        )
        routing.update(
            {
                "weights": weights,
                "selected_mask": selected,
                "selected_indices": indices,
                "weight_normalization": self.weight_normalization,
                "straight_through": self.straight_through,
            }
        )
        return routing


def _translate_with_fill(
    value: torch.Tensor,
    shift_y: int,
    shift_x: int,
    *,
    fill_value: float | complex,
) -> torch.Tensor:
    """Translate the last two axes without circular wrap-around."""

    shift_y = int(shift_y)
    shift_x = int(shift_x)
    shifted = torch.roll(value, (shift_y, shift_x), dims=(-2, -1))
    height, width = value.shape[-2:]
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return torch.full_like(value, fill_value)
    if shift_y > 0:
        shifted[..., :shift_y, :] = fill_value
    elif shift_y < 0:
        shifted[..., shift_y:, :] = fill_value
    if shift_x > 0:
        shifted[..., :, :shift_x] = fill_value
    elif shift_x < 0:
        shifted[..., :, shift_x:] = fill_value
    return shifted


def _sample_shift(maximum: int, *, training: bool) -> tuple[int, int]:
    maximum = int(maximum)
    if not training or maximum <= 0:
        return 0, 0
    return (
        int(torch.randint(-maximum, maximum + 1, ()).item()),
        int(torch.randint(-maximum, maximum + 1, ()).item()),
    )


class OpticalDetectorTopKRouter(nn.Module):
    """Phase-only, time-multiplexed optical top-k router.

    The feature amplitude is placed once at the centre of the unchanged
    numerical canvas.  A learned phase-only mask and the same 10 cm angular
    spectrum propagation produce one CCD frame.  Four fixed detector windows
    map raw optical energy to four expert scores.  Only normalization,
    softmax and top-k selection are electronic.
    """

    implementation_name = "phase_only_detector_energy_topk"

    def __init__(self, geometry: Any, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        self.num_experts = int(geometry.num_experts)
        self.top_k = int(settings.top_k)
        self.temperature = float(settings.router_temperature)
        self.noise_std = float(settings.router_noise_std)
        self.eps = float(settings.optical_router_energy_eps)
        self.input_size = int(geometry.expert_size)
        self.canvas_size = int(geometry.canvas_size)
        self.active_size = int(geometry.active_size)
        self.input_shift_pixels = int(settings.optical_router_input_shift_pixels)
        self.phase_shift_pixels = int(settings.optical_router_phase_shift_pixels)
        self.ccd_shift_pixels = int(settings.optical_router_ccd_shift_pixels)
        self.phase_dropout_p = float(settings.optical_router_phase_dropout_p)
        self.phase_dropout_block_size = int(
            settings.optical_router_phase_dropout_block_size
        )
        self.capture_loss_scale = float(settings.optical_router_capture_loss_scale)
        self.weight_normalization = str(settings.router_weight_normalization)
        self.straight_through = bool(settings.router_straight_through)
        self.score_normalization = str(settings.optical_router_score_normalization)
        self.logical_pixel_pitch_um = float(settings.language_optical_pixel_pitch_um)
        self.distance_m = float(settings.language_optical_distance_m)
        self.wavelength_nm = float(settings.language_optical_wavelength_nm)

        intervals = tuple(tuple(map(int, item)) for item in settings.optical_router_detector_intervals)
        if len(intervals) != 2:
            raise ValueError("Optical router requires two detector intervals per axis")
        masks = torch.zeros(
            self.num_experts, self.active_size, self.active_size, dtype=torch.float32
        )
        index = 0
        bounds: list[tuple[int, int, int, int]] = []
        for top, bottom in intervals:
            for left, right in intervals:
                masks[index, top:bottom, left:right] = 1.0
                bounds.append((left, top, right, bottom))
                index += 1
        if index != self.num_experts:
            raise ValueError("Detector grid must define exactly four expert regions")
        self.detector_bounds = tuple(bounds)
        self.register_buffer("detector_masks", masks, persistent=False)

        initial_phase = self._four_spot_initial_phase(intervals)
        normalized = (initial_phase / (2.0 * math.pi)).clamp(1.0e-4, 1.0 - 1.0e-4)
        # Deliberately avoid the generic ``raw_phase`` name: the shared
        # optimizer assigns every such tensor to the four feature-mask phase
        # group.  Router phase owns the independent router learning rate.
        self.raw_router_phase = nn.Parameter(torch.logit(normalized).float())
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=self.wavelength_nm * 1.0e-9,
            pixel_size_m=self.logical_pixel_pitch_um * 1.0e-6,
            distance_m=self.distance_m,
            grid_size=self.canvas_size,
            k_space_constraint_enabled=bool(settings.language_optical_k_space_enabled),
            theta_max_deg=float(settings.language_optical_theta_max_deg),
        )

        self.last_input_amplitude: torch.Tensor | None = None
        self.last_detector_intensity: torch.Tensor | None = None
        self.last_detector_energy: torch.Tensor | None = None
        self.last_capture_fraction: torch.Tensor | None = None
        self.last_shifts: dict[str, tuple[int, int]] = {}
        self._measured_routing: dict[str, torch.Tensor] | None = None

    def set_measured_routing(
        self, payload: Mapping[str, torch.Tensor] | None
    ) -> None:
        """Install an explicit batch-aligned hardware routing decision.

        This is intentionally a routing contract rather than a measured CCD
        setter: the hardware bridge rectifies the CCD to canonical 478x478,
        integrates the four audited windows and records the resulting routing
        manifest before injecting these four tensors.  ``None`` restores the
        differentiable optical simulation.
        """

        if payload is None:
            self._measured_routing = None
            return
        required = {
            "probabilities",
            "weights",
            "selected_mask",
            "selected_indices",
        }
        missing = required.difference(payload)
        unexpected = set(payload).difference(required)
        if missing or unexpected:
            raise ValueError(
                "Measured routing payload keys mismatch: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        self._measured_routing = {
            key: torch.as_tensor(payload[key]).detach().cpu().clone()
            for key in sorted(required)
        }

    def _measured_forward(
        self, input_fields: torch.Tensor
    ) -> dict[str, torch.Tensor | str | bool]:
        if self._measured_routing is None:
            raise RuntimeError("No measured routing payload is installed")
        batch = len(input_fields)
        device = input_fields.device
        dtype = input_fields.dtype
        probabilities = self._measured_routing["probabilities"].to(
            device=device, dtype=dtype
        )
        weights = self._measured_routing["weights"].to(device=device, dtype=dtype)
        selected = self._measured_routing["selected_mask"].to(
            device=device, dtype=torch.bool
        )
        indices = self._measured_routing["selected_indices"].to(
            device=device, dtype=torch.long
        )
        expected = (batch, self.num_experts)
        if tuple(probabilities.shape) != expected or tuple(weights.shape) != expected:
            raise ValueError(
                f"Measured probabilities/weights must both be {expected}, got "
                f"{tuple(probabilities.shape)} and {tuple(weights.shape)}"
            )
        if tuple(selected.shape) != expected or tuple(indices.shape) != (
            batch,
            self.top_k,
        ):
            raise ValueError(
                "Measured selected_mask/selected_indices shape mismatch for "
                f"batch={batch}, top_k={self.top_k}"
            )
        for label, value in (("probabilities", probabilities), ("weights", weights)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Measured routing {label} contains non-finite values")
            if bool((value < 0.0).any()):
                raise ValueError(f"Measured routing {label} must be nonnegative")
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(batch, device=device, dtype=dtype),
            rtol=1.0e-4,
            atol=1.0e-5,
            msg="Measured routing probabilities must sum to one",
        )
        reconstructed_mask = torch.zeros_like(selected).scatter(1, indices, True)
        if not torch.equal(selected, reconstructed_mask):
            raise ValueError(
                "Measured selected_indices do not encode the supplied selected_mask"
            )
        if not bool((selected.sum(dim=-1) == self.top_k).all()):
            raise ValueError("Measured routing does not select exactly top_k experts")
        if bool((weights.masked_select(~selected).abs() > 1.0e-7).any()):
            raise ValueError("Measured unselected expert weights must be zero")
        if self.weight_normalization == "legacy_l1":
            norm = weights.sum(dim=-1)
        else:
            norm = weights.square().sum(dim=-1).sqrt()
        torch.testing.assert_close(
            norm,
            torch.ones_like(norm),
            rtol=1.0e-4,
            atol=1.0e-5,
            msg="Measured routing weights violate the configured normalization",
        )

        importance = probabilities.mean(dim=0)
        load = selected.float().mean(dim=0) / float(self.top_k)
        balance = float(self.num_experts) * torch.sum(importance * load)
        importance_loss = (
            float(self.num_experts) * torch.sum(importance.square()) - 1.0
        )
        entropy = -(
            probabilities.clamp_min(self.eps).log() * probabilities
        ).sum(dim=-1).mean() / math.log(float(self.num_experts))
        energy = probabilities
        captured = torch.ones(batch, device=device, dtype=dtype)
        self.last_detector_energy = energy.detach()
        self.last_capture_fraction = captured.detach()
        self.last_detector_intensity = None
        return {
            "logits": probabilities.clamp_min(self.eps).log() * self.temperature,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "balance_loss": balance,
            "load_balance_loss": balance,
            "importance_loss": importance_loss,
            "normalized_entropy": entropy,
            "importance": importance,
            "load": load,
            "detector_energy": energy,
            "detector_energy_fraction": probabilities,
            "capture_fraction": captured,
            "capture_loss": input_fields.new_zeros(()),
            "capture_loss_scale": input_fields.new_tensor(self.capture_loss_scale),
            "weight_normalization": self.weight_normalization,
            "straight_through": False,
            "score_normalization": "precomputed_from_measured_router_ccd",
            "router_implementation": self.implementation_name,
            "phase_prompt_used": True,
            "amplitude_phase_relay": "measured_router_manifest",
            "measured_routing": True,
        }

    def _four_spot_initial_phase(
        self, intervals: tuple[tuple[int, int], tuple[int, int]]
    ) -> torch.Tensor:
        """Initialize a phase-only hologram whose four orders hit the ROIs."""

        detector_center = 0.5 * (self.active_size - 1)
        centers = [0.5 * (left + right - 1) for left, right in intervals]
        coordinates = (
            torch.arange(self.input_size, dtype=torch.float64)
            - 0.5 * (self.input_size - 1)
        ) * (self.logical_pixel_pitch_um * 1.0e-6)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        wavelength_m = self.wavelength_nm * 1.0e-9
        pitch_m = self.logical_pixel_pitch_um * 1.0e-6
        phasors = torch.zeros_like(xx, dtype=torch.complex128)
        for target_y in centers:
            for target_x in centers:
                offset_x_m = (target_x - detector_center) * pitch_m
                offset_y_m = (target_y - detector_center) * pitch_m
                frequency_x = offset_x_m / (wavelength_m * self.distance_m)
                frequency_y = offset_y_m / (wavelength_m * self.distance_m)
                angle = 2.0 * math.pi * (frequency_x * xx + frequency_y * yy)
                phasors = phasors + torch.exp(1j * angle)
        return torch.remainder(torch.angle(phasors), 2.0 * math.pi).float()

    def phase(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sigmoid(self.raw_router_phase)

    def active_phase(self) -> torch.Tensor:
        """Return the 478x478 logical phase payload used for hardware export."""

        value = self.phase()
        margin = (self.active_size - self.input_size) // 2
        return F.pad(
            value,
            (
                margin,
                self.active_size - self.input_size - margin,
                margin,
                self.active_size - self.input_size - margin,
            ),
        )

    def set_noise_std(self, value: float) -> None:
        self.noise_std = max(0.0, float(value))

    def _phase_modulation(self, batch: int) -> torch.Tensor:
        phase = self.phase().unsqueeze(0).expand(batch, -1, -1)
        if self.training and self.phase_dropout_p > 0.0:
            block = self.phase_dropout_block_size
            coarse_h = int(math.ceil(self.input_size / block))
            coarse_w = int(math.ceil(self.input_size / block))
            bypass = torch.rand(
                batch, 1, coarse_h, coarse_w, device=phase.device
            ) < self.phase_dropout_p
            bypass = F.interpolate(
                bypass.float(), size=(self.input_size, self.input_size), mode="nearest"
            )[:, 0].bool()
            phase = torch.where(bypass, torch.zeros_like(phase), phase)
        return torch.exp(1j * phase).to(torch.complex64)

    def _simulate(self, fields: torch.Tensor) -> torch.Tensor:
        batch = len(fields)
        input_shift = _sample_shift(self.input_shift_pixels, training=self.training)
        phase_shift = _sample_shift(self.phase_shift_pixels, training=self.training)
        ccd_shift = _sample_shift(self.ccd_shift_pixels, training=self.training)
        self.last_shifts = {
            "input": input_shift,
            "phase": phase_shift,
            "ccd": ccd_shift,
        }

        input_margin = (self.canvas_size - self.input_size) // 2
        input_canvas = F.pad(
            fields.float(),
            (
                input_margin,
                self.canvas_size - self.input_size - input_margin,
                input_margin,
                self.canvas_size - self.input_size - input_margin,
            ),
        )
        input_canvas = _translate_with_fill(
            input_canvas, *input_shift, fill_value=0.0
        )
        phase_canvas = torch.ones_like(input_canvas, dtype=torch.complex64)
        modulation = self._phase_modulation(batch)
        phase_canvas[
            :,
            input_margin : input_margin + self.input_size,
            input_margin : input_margin + self.input_size,
        ] = modulation
        phase_canvas = _translate_with_fill(
            phase_canvas, *phase_shift, fill_value=1.0 + 0.0j
        )
        detector_field = self.propagator(input_canvas.to(torch.complex64) * phase_canvas)
        full_intensity = detector_field.abs().square().float()
        full_intensity = _translate_with_fill(
            full_intensity, *ccd_shift, fill_value=0.0
        )
        active = self.geometry.active_aperture
        intensity = full_intensity[
            :, active.y0 : active.y1, active.x0 : active.x1
        ]
        self.last_input_amplitude = input_canvas.detach()
        self.last_detector_intensity = intensity.detach()
        return intensity

    def forward(self, input_fields: torch.Tensor) -> dict[str, torch.Tensor | str | bool]:
        if input_fields.ndim != 3 or tuple(input_fields.shape[-2:]) != (
            self.input_size,
            self.input_size,
        ):
            raise ValueError(
                f"Optical router input must be [B,{self.input_size},{self.input_size}], "
                f"got {tuple(input_fields.shape)}"
            )
        if self._measured_routing is not None:
            return self._measured_forward(input_fields)
        intensity = self._simulate(input_fields)
        energy = torch.einsum("bhw,ehw->be", intensity, self.detector_masks)
        total_energy = intensity.sum(dim=(-2, -1)).clamp_min(self.eps)
        captured = (energy.sum(dim=-1) / total_energy).clamp(0.0, 1.0)
        energy_fraction = energy / energy.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if self.score_normalization == "standardized_region_energy":
            centered = energy - energy.mean(dim=-1, keepdim=True)
            logits = centered / centered.square().mean(
                dim=-1, keepdim=True
            ).add(self.eps).sqrt()
        elif self.score_normalization == "log_energy_fraction":
            logits = energy_fraction.clamp_min(self.eps).log()
        else:
            raise RuntimeError(
                f"Unsupported optical router score normalization {self.score_normalization!r}"
            )
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        weights, selected, indices = sparsify_probabilities(
            probabilities,
            self.top_k,
            normalization=self.weight_normalization,
            straight_through=self.straight_through,
            eps=self.eps,
        )
        importance = probabilities.mean(dim=0)
        load = selected.float().mean(dim=0) / float(self.top_k)
        balance = float(self.num_experts) * torch.sum(importance * load)
        importance_loss = float(self.num_experts) * torch.sum(importance.square()) - 1.0
        entropy = -(
            probabilities.clamp_min(self.eps).log() * probabilities
        ).sum(dim=-1).mean() / math.log(float(self.num_experts))
        capture_loss = (1.0 - captured).mean()
        # The shared trainer has two router loss slots. Keep its exact public
        # contract and attach the optical-efficiency term to balance_loss with
        # an explicit, reported internal scale.
        balance_with_capture = balance + self.capture_loss_scale * capture_loss
        self.last_detector_energy = energy.detach()
        self.last_capture_fraction = captured.detach()
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "balance_loss": balance_with_capture,
            "load_balance_loss": balance,
            "importance_loss": importance_loss,
            "normalized_entropy": entropy,
            "importance": importance,
            "load": load,
            "detector_energy": energy,
            "detector_energy_fraction": energy_fraction,
            "capture_fraction": captured,
            "capture_loss": capture_loss,
            "capture_loss_scale": input_fields.new_tensor(self.capture_loss_scale),
            "weight_normalization": self.weight_normalization,
            "straight_through": self.straight_through,
            "score_normalization": self.score_normalization,
            "router_implementation": self.implementation_name,
            "phase_prompt_used": True,
            "amplitude_phase_relay": "same_co_planar_4f_then_10cm_router_exposure",
        }


__all__ = [
    "FairElectronicAmplitudeRouter",
    "OpticalDetectorTopKRouter",
    "ROUTING_NORMALIZATIONS",
    "sparsify_probabilities",
]
