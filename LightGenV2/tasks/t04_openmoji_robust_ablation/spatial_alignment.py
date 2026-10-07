"""Training-only spatial field perturbations; no inference-layer additions."""
import math
import torch
import torch.nn.functional as F


def perturb_field(x, shift_pixels=1., rotation_degrees=.25, dropout=.02):
    """Shared geometry per optical call/batch; zero occlusion, no rescaling.

    Pixel units refer to the simulation field, not measured camera pixels.
    These bounds are experimental hypotheses, not geometric calibration.
    """
    assert shift_pixels >= 0 and rotation_degrees >= 0 and 0 <= dropout < 1
    if shift_pixels == rotation_degrees == dropout == 0:
        return x
    h, w = x.shape[-2:]
    real = x.real if x.is_complex() else x
    theta = real.new_zeros((1, 2, 3))
    angle = real.new_empty(()).uniform_(-rotation_degrees, rotation_degrees) * math.pi / 180
    theta[0, 0, 0] = theta[0, 1, 1] = angle.cos()
    theta[0, 0, 1] = -angle.sin()
    theta[0, 1, 0] = angle.sin()
    theta[0, 0, 2] = real.new_empty(()).uniform_(-shift_pixels, shift_pixels) * 2 / w
    theta[0, 1, 2] = real.new_empty(()).uniform_(-shift_pixels, shift_pixels) * 2 / h
    grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
    def warp(v):
        a = v.reshape(-1, 1, h, w)
        return F.grid_sample(a, grid.expand(len(a), -1, -1, -1),
                             mode='bilinear', padding_mode='zeros',
                             align_corners=False).reshape(v.shape)
    y = (torch.complex(warp(x.real), warp(x.imag)) if x.is_complex() else warp(x)) if (shift_pixels or rotation_degrees) else x
    if dropout:
        # One coarse spatial occlusion map per field frame, shared real/imag.
        mask = (torch.rand((1, 1, 16, 16), device=x.device) >= dropout).to(real.dtype)
        mask = F.interpolate(mask, size=(h, w), mode='nearest').reshape(h, w)
        y = y * mask
    return y
