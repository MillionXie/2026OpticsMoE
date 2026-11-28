"""Footprint-constrained selection on the optical router's continuous face."""
import torch
from torch.nn import functional as F

from .model import SpatialRoutingLG


class FootprintRoutingLG(SpatialRoutingLG):
    """Two bright separated placements. Overlap is relative to one 224² copy."""

    max_overlap_fraction = 0.25
    side = 224

    def __init__(self, num_classes=8):
        super().__init__(mode="continuous", num_classes=num_classes)
        self.mode = "constrained"
        coefficients = torch.tensor([1.0, 2.0, 1.0]) / 4
        self.register_buffer("binomial", torch.outer(coefficients, coefficients)[None, None],
                             persistent=False)

    def _candidate_positions(self, scores):
        # Replicate at the active CCD search boundary for ranking. For centroid
        # refinement, only measured neighbors enter the local weighted sums.
        kernel = self.binomial.to(scores)
        smoothed = F.conv2d(F.pad(scores[:, None], (1, 1, 1, 1), mode="replicate"), kernel)
        grid = self.anchor_yx.reshape(len(self.axis), len(self.axis), 2).permute(2, 0, 1)
        grid = grid.to(scores)
        weight = smoothed + 1e-12
        totals = F.conv2d(weight, kernel, padding=1)
        moments = F.conv2d(weight * grid, kernel.expand(2, 1, 3, 3), padding=1, groups=2)
        positions = (moments / totals).clamp(0, 254)
        # Keep edges on the physical boundary along the outward normal. This
        # retains all four corners, guaranteeing at least a disjoint pair.
        positions = torch.where((grid == 0) | (grid == 254), grid, positions)
        return positions.flatten(2).transpose(1, 2), smoothed.flatten(1)

    @torch.no_grad()
    def _select_pair(self, positions, smooth_power):
        # Hard selection is outside autograd, as with the original peak indices.
        # Maximize a joint score over *all* feasible pairs. Using log powers
        # favors two usable detector readings even on a broad dominant peak.
        y, x = positions.unbind(-1)
        shared_height = (self.side - (y[:, :, None] - y[:, None, :]).abs()).clamp_min_(0)
        shared_width = (self.side - (x[:, :, None] - x[:, None, :]).abs()).clamp_min_(0)
        overlap_area = shared_height.mul_(shared_width)
        log_power = (smooth_power + 1e-12).log()
        joint_score = log_power[:, :, None] + log_power[:, None, :]
        joint_score.masked_fill_(overlap_area > self.max_overlap_fraction * self.side**2, -torch.inf)
        n = len(self.anchor_yx)
        pair = joint_score.flatten(1).argmax(1)
        first, second = pair // n, pair % n
        batch = torch.arange(len(positions), device=positions.device)
        swap = smooth_power[batch, second] > smooth_power[batch, first]
        return torch.stack((torch.where(swap, second, first), torch.where(swap, first, second)), 1)

    def continuous_coordinates(self, scores):
        positions, smooth_power = self._candidate_positions(scores)
        indices = self._select_pair(positions, smooth_power)
        selected_positions = positions.gather(1, indices[..., None].expand(-1, -1, 2))
        # Allocate light using the actual 60² CCD readings at both selected
        # anchors; smoothed values determine the addresses only.
        return selected_positions, scores.flatten(1).gather(1, indices)

    def route_from_ccd(self, intensity):
        """Replay routing for a measured or synthetically perturbed router CCD."""
        if intensity.ndim != 3 or intensity.shape[-2:] != (518, 518):
            raise ValueError("Expected router CCD intensity of shape Bx518x518")
        positions, selected = self.continuous_coordinates(self.window_scores(intensity))
        capture = selected.sum(1)
        return positions, (selected + 1e-12) / (capture[:, None] + 2e-12), capture
