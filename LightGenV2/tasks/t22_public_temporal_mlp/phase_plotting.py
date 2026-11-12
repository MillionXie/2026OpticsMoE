"""Shared monotone phase color normalization with an explicit physical scale."""
from __future__ import annotations

import math
from matplotlib.colors import FuncNorm
import numpy as np


PHASE_DISPLAY={'kind':'asinh','center_rad':math.pi,'scale_rad':.25,
               'minimum_rad':0.,'maximum_rad':2*math.pi}


def phase_display_norm():
    center=PHASE_DISPLAY['center_rad']
    scale=PHASE_DISPLAY['scale_rad']
    return FuncNorm((lambda value:np.arcsinh((value-center)/scale),
                     lambda value:center+scale*np.sinh(value)),
                    vmin=0,vmax=2*math.pi)


def add_phase_colorbar(fig,artist,ax):
    bar=fig.colorbar(artist,ax=ax,shrink=.6,aspect=14,pad=.03,
                    label='Phase (rad)',ticks=[0,math.pi,2*math.pi])
    bar.ax.set_yticklabels(['0','π','2π'])
    return bar
