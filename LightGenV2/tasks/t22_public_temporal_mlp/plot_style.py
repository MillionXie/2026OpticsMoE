"""Shared Arial typography for the current model's scientific figures."""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.text import Text


FONT_FAMILY = 'Arial'
BODY_SIZE_PT = 7
TITLE_SIZE_PT = 8

mpl.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': BODY_SIZE_PT,
    'axes.labelsize': BODY_SIZE_PT,
    'axes.titlesize': TITLE_SIZE_PT,
    'xtick.labelsize': BODY_SIZE_PT,
    'ytick.labelsize': BODY_SIZE_PT,
    'legend.fontsize': BODY_SIZE_PT,
    'figure.titlesize': TITLE_SIZE_PT,
    'mathtext.fontset': 'custom',
    'mathtext.rm': FONT_FAMILY,
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.sf': FONT_FAMILY,
    'mathtext.tt': FONT_FAMILY,
    'mathtext.default': 'regular',
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.9,
    'lines.markersize': 3,
})


def save_figure(figure, path):
    """Apply typography to existing text and save a compact 300-dpi figure."""
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=FONT_FAMILY),
        fallback_to_default=False,
    )
    for text in figure.findobj(Text):
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(BODY_SIZE_PT)
    for axis in figure.axes:
        axis.title.set_fontsize(TITLE_SIZE_PT)
    if figure._suptitle is not None:
        figure._suptitle.set_fontsize(TITLE_SIZE_PT)
    figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.06)
    return {
        'font_family': FONT_FAMILY,
        'font_file': str(Path(font_path).resolve()),
        'body_size_pt': BODY_SIZE_PT,
        'title_size_pt': TITLE_SIZE_PT,
        'dpi': 300,
        'figure_size_inches': figure.get_size_inches().tolist(),
    }
