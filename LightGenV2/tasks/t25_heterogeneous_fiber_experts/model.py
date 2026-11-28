"""Four fixed optical experts with an optional coherent single-mode fiber array."""
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t24_continuous_spatial_routing.model import SpatialRoutingLG


SLOTS_YX = ((0, 0), (0, 254), (254, 0), (254, 254))
SLOT_SIDE = 224
CELL_SIDE = 8
WAIST_PX = 2.0


def lp01_mode(dtype=torch.float32):
    """Unit-norm, cell-centred Gaussian. Waist is the 1/e field-amplitude radius."""
    x = torch.arange(CELL_SIDE, dtype=dtype) - (CELL_SIDE - 1) / 2
    u = torch.exp(-(x[:, None].square() + x.square()) / WAIST_PX**2)
    return u / torch.linalg.vector_norm(u)


def project_fiber_cells(field, mode):
    """Apply |u><u| separately in each disjoint cell of the complex entrance field."""
    if not field.is_complex() or field.shape[-2] % CELL_SIDE or field.shape[-1] % CELL_SIDE:
        raise ValueError("Fiber projection requires a complex field tiled by complete 8x8 cells")
    if mode.shape != (CELL_SIDE, CELL_SIDE) or mode.is_complex():
        raise ValueError("Expected the real 8x8 LP01 basis")
    height, width = field.shape[-2:]
    cells = field.reshape(*field.shape[:-2], height // CELL_SIDE, CELL_SIDE, width // CELL_SIDE, CELL_SIDE)
    u = mode.to(dtype=field.real.dtype)[None, :, None, :]
    # The real mode is its own conjugate. Each coefficient retains its complex phase.
    coefficients = (cells * u).sum(dim=(-3, -1), keepdim=True)
    return (coefficients * u).reshape_as(field)


class HeterogeneousFiberLG(SpatialRoutingLG):
    """T24 fixed Top-2, with ideal equal-phase, unit-amplitude core transfer at selected slots."""

    def __init__(self, arm, num_classes=8, *, fiber_slots=None):
        super().__init__(mode="fixed", num_classes=num_classes)
        if arm not in {"diffraction", "mixed", "fiber", "uniform_half"}:
            raise ValueError(f"Unknown expert arm: {arm}")
        if fiber_slots is None:
            fiber_slots = (0, 1, 2, 3) if arm in {"fiber", "uniform_half"} else ()
        fiber_slots = tuple(fiber_slots)
        if any(type(i) is not int for i in fiber_slots):
            raise ValueError("Fiber slot indices must be integers")
        slots = tuple(sorted(fiber_slots))
        choices = {"diffraction": ((),), "mixed": ((0, 3), (1, 2)),
                   "fiber": ((0, 1, 2, 3),), "uniform_half": ((0, 1, 2, 3),)}
        if slots not in choices[arm]:
            raise ValueError(f"Invalid fiber slots {slots} for {arm}")
        self.arm = arm
        self.fiber_slots = slots
        n = SLOT_SIDE // CELL_SIDE
        cell_mask = torch.zeros((4, n, n), dtype=torch.bool)
        axis = torch.arange(n)
        checkerboard = (axis[:, None] + axis[None, :]) % 2 == 0
        for slot in slots:
            cell_mask[slot] = checkerboard if arm == "uniform_half" else True
        self.register_buffer("fiber_cell_mask", cell_mask, persistent=False)
        self.fiber_core_counts = tuple(cell_mask.sum(dim=(1, 2)).tolist())
        pixels = torch.arange(478**2).reshape(478, 478)
        tiles = []
        for slot in slots:
            y, x = SLOTS_YX[slot]
            tile = pixels[y:y+SLOT_SIDE, x:x+SLOT_SIDE]
            cells = tile.reshape(n, CELL_SIDE, n, CELL_SIDE).permute(0, 2, 1, 3)
            tiles.append(cells[cell_mask[slot]].reshape(-1))
        indices = torch.cat(tiles) if tiles else torch.empty(0, dtype=torch.long)
        self.register_buffer("fiber_indices", indices, persistent=False)
        self.register_buffer("core_mode", lp01_mode(), persistent=False)

    def filter_experts(self, field):
        """Filter only fiber slots on the shared 478² face, before free-space propagation."""
        if not self.fiber_slots:
            return field
        flat = field.flatten(1)
        cells = flat.index_select(1, self.fiber_indices).reshape(len(field), -1, CELL_SIDE, CELL_SIDE)
        transmitted = project_fiber_cells(cells, self.core_mode)
        return flat.index_copy(1, self.fiber_indices, transmitted.flatten(1)).reshape_as(field)

    @staticmethod
    def slot_power(field):
        return torch.stack([
            field[:, y:y+SLOT_SIDE, x:x+SLOT_SIDE].abs().square().sum((-2, -1)) for y, x in SLOTS_YX
        ], dim=1)

    def diagnostic_route(self, amplitude):
        # Same arithmetic and tie convention as T24 fixed locate; also retain
        # its four pre-selection detector integrals from this acquisition.
        field = amplitude * self.transmission(self.router_phase)
        intensity = self.propagator(F.pad(field, (147,) * 4)).abs().square()
        energy = self.detect(intensity, self.router_centers, 60)
        dense = (energy + 1e-12) / (energy.sum(1, keepdim=True) + 4e-12)
        slots = dense.topk(2, dim=1).indices
        selected = energy.gather(1, slots)
        capture = selected.sum(1)
        power = (selected + 1e-12) / (capture[:, None] + 2e-12)
        full_gate = torch.zeros_like(energy).scatter(1, slots, power)
        details = dict(router_port_energy=energy, dense_power=dense, selected_slots=slots, slot_gate_power=full_gate)
        return self.corners[slots].to(amplitude.dtype), power, capture, details

    def forward(self, amplitude, *, diagnostics=False):
        if amplitude.ndim != 3 or amplitude.shape[-2:] != (224, 224):
            raise ValueError("Expected encoded Bx224x224 amplitudes")
        amplitude = amplitude / amplitude.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)
        if diagnostics:
            positions, power, capture, details = self.diagnostic_route(amplitude)
        else:
            positions, power, capture = self.locate(amplitude)
            details = {}
        entrance = self.multiplex(amplitude, positions, power)
        modulated = entrance * self.transmission(self.first_phase)
        transmitted = self.filter_experts(modulated)
        if diagnostics:
            before = self.slot_power(modulated)
            details.update(pre_fiber_power=before,
                           transmitted_power=self.slot_power(transmitted) if self.fiber_slots else before)
        first = F.pad(transmitted, (20,) * 4)
        field = self.oeo(self.propagator(first))
        global_mask = F.pad(self.transmission(self.global_phase), (20,) * 4, value=1)
        field = self.oeo(self.propagator(field * global_mask))
        active = field[:, 20:498, 20:498].abs().square()
        features = F.adaptive_avg_pool2d(active[:, None], 28).flatten(1)
        features = features / features.mean(1, keepdim=True).clamp_min(1e-20)
        return dict(logits=self.shared_head(features), ccd_features=features,
                    positions=positions, power=power, router_capture=capture, **details)
