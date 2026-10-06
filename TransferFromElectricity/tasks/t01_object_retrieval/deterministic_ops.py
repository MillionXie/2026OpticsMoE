"""Task-local deterministic adjoint for adaptive average pooling.

Native CUDA pool forward is retained. Its overlapping-bin atomic backward is
replaced by the same linear map's separable matrix transpose. No optics changes.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F

_pool1d, _pool2d = F.adaptive_avg_pool1d, F.adaptive_avg_pool2d


@lru_cache(maxsize=64)
def _weights(inputs, outputs, device, dtype):
    result = torch.zeros(outputs, inputs, device=device, dtype=dtype)
    for index in range(outputs):
        start = index * inputs // outputs
        stop = ((index + 1) * inputs + outputs - 1) // outputs
        result[index, start:stop] = 1. / (stop - start)
    return result


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, output_size, dimensions):
        result = (_pool1d if dimensions == 1 else _pool2d)(value, output_size)
        ctx.input_shape = value.shape
        ctx.dimensions = dimensions
        return result

    @staticmethod
    def backward(ctx, grad):
        with torch.autocast(grad.device.type, enabled=False):
            wx = _weights(ctx.input_shape[-1], grad.shape[-1], grad.device, grad.dtype)
            result = grad @ wx
            if ctx.dimensions == 2:
                wy = _weights(ctx.input_shape[-2], grad.shape[-2], grad.device, grad.dtype)
                result = wy.T @ result
        return result, None, None


def adaptive_avg_pool1d(value, output_size):
    return _Pool.apply(value, output_size, 1)


def adaptive_avg_pool2d(value, output_size):
    return _Pool.apply(value, output_size, 2)


def install_deterministic_pooling():
    """Opt-in within this training process; inherited source files stay intact."""
    F.adaptive_avg_pool1d = adaptive_avg_pool1d
    F.adaptive_avg_pool2d = adaptive_avg_pool2d
