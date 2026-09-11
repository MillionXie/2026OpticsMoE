"""Panel-specific gray LUT direction, independent of spatial orientation.

Legacy/sister panels default to normal. Inverted files live in a NEW tree;
existing sessions and their phase hashes are never silently rewritten.
"""
from pathlib import Path
import numpy as np


def mode(c):
    value=c['phase_slm'].get('gray_encoding','normal')
    if value not in ('normal','inverted_255_minus_g'):raise ValueError('Unknown phase gray_encoding: '+value)
    return value


def encode_gray(image,c):
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=2:raise ValueError('Phase encoding requires 2D uint8 BMP raster')
    return 255-a if mode(c)=='inverted_255_minus_g' else a


def generated_root(root,c):
    return Path(root)/'generated'/('phase_inverted' if mode(c)=='inverted_255_minus_g' else '')
