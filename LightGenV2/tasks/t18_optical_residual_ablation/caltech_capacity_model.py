"""Spatial phase degrees-of-freedom probe; original optical apertures unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from caltech_repair234_model import CorrectedMoE, input_amplitude
from caltech_repair234_model import deployment_contract as repaired_contract
from caltech_deploy234_model import APERTURES

GRIDS = ('full', '128', '64')
COUNTS = {'full': 908552, '128': 214016, '64': 91136}

class CapacityMoE(CorrectedMoE):
    def __init__(self, depth=4, rho=.3, grid='full'):
        assert depth == 4 and rho == .3 and grid in GRIDS
        super().__init__(depth, rho)
        self.phase_grid = grid
        # Router uses the same seed17 draw as the frozen graph. All new main
        # masks start at raw0, giving identical physical phi=pi across shapes.
        self.physical_shapes = [tuple(p.shape) for p in self.phase_sets]
        self.phase_sets = nn.ParameterList([
            nn.Parameter(torch.zeros(shape if grid == 'full' else (int(grid), int(grid))))
            for shape in self.physical_shapes])
        assert sum(p.numel() for p in self.parameters()) == COUNTS[grid]

    def expanded_raw(self, index):
        raw = self.phase_sets[index]
        size = self.physical_shapes[index]
        return raw if tuple(raw.shape) == size else F.interpolate(
            raw[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]

    def plane(self, stage):
        if stage == 'router':
            return super().plane(stage)
        index = int(stage)
        plane = torch.ones(518, 518, dtype=torch.complex64, device=self.raw_router.device)
        lo, hi = self.stage_slices[index]
        locations = APERTURES if self.stages[index] == 'expert' else [(20, 20)]
        for j, (y, x) in zip(range(lo, hi), locations):
            raw = self.expanded_raw(j)
            modulation = (1-self.rho)*torch.exp(2j*math.pi*torch.sigmoid(raw))+self.rho
            h, w = self.physical_shapes[j]
            plane[y:y+h, x:x+w] = modulation
        return plane

    def smoothness(self):
        # Same 22 physical phase-neighbour terms as the full-resolution model.
        terms = []
        for raw in [self.raw_router] + [self.expanded_raw(j) for j in range(10)]:
            phase = 2*math.pi*torch.sigmoid(raw)
            terms.extend([(phase[1:]-phase[:-1]).square().mean(),
                          (phase[:, 1:]-phase[:, :-1]).square().mean()])
        return torch.stack(terms).mean()

def deployment_contract(depth, rho, grid='full'):
    value = repaired_contract(depth, rho)
    value.update(phase_grid=grid, parameterization='bilinear raw phase grid to original224/478, align_corners=False, then sigmoid/M/8bit',
                 main_initialization='raw0, physical phi=pi and M=-.4; shared new control',
                 phase_smoothness='22 equally weighted physical expanded phase-neighbour terms',
                 architecture_status='Diagnostic parameter constraint only; does not adopt a smaller final network')
    return value
