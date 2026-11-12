"""Linear display of the physical phase on its principal-value interval."""
from __future__ import annotations

import math
from matplotlib.colors import Normalize
import numpy as np


PHASE_DISPLAY={'kind':'linear','minimum_rad':-math.pi,'maximum_rad':math.pi,
               'wrapping':'atan2(sin(phase), cos(phase))'}


def phase_display_values(value):
    return np.arctan2(np.sin(value),np.cos(value))


def phase_display_norm():
    return Normalize(vmin=-math.pi,vmax=math.pi)


def add_phase_colorbar(fig,artist,ax,label='Phase (rad)'):
    bar=fig.colorbar(artist,ax=ax,shrink=.6,aspect=14,pad=.03,
                    label=label,ticks=[-math.pi,0,math.pi])
    bar.ax.set_yticklabels(['−π','0','π'])
    return bar
