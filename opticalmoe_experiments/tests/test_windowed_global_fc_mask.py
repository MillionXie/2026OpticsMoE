import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.expert_phase_layer import GlobalFCPhaseMask


def test_windowed_global_fc_uses_center_window_only():
    fc = GlobalFCPhaseMask(
        canvas_shape=(520, 520),
        phase_size=450,
        phase_mode="center_window",
        phase_init="identity",
    )
    assert fc.phase.raw_phase.shape == (450, 450)
    assert fc.phase_region() == [35, 485, 35, 485]
    assert fc.trainable_parameter_count() == 450 * 450
    assert fc.get_phase_wrapped().shape == (450, 450)
    assert fc.get_phase_canvas_wrapped().shape == (520, 520)


def test_windowed_global_fc_forward_keeps_padding_transparent():
    fc = GlobalFCPhaseMask(
        canvas_shape=(12, 12),
        phase_size=6,
        phase_mode="center_window",
        phase_init="identity",
    )
    with torch.no_grad():
        fc.phase.raw_phase.fill_(math.pi / 2.0)
    field = torch.ones(1, 12, 12, dtype=torch.complex64)
    out = fc(field)
    y0, y1, x0, x1 = fc.phase_region()
    outside_mask = torch.ones(12, 12, dtype=torch.bool)
    outside_mask[y0:y1, x0:x1] = False
    assert torch.allclose(out[0][outside_mask], field[0][outside_mask])
    assert torch.allclose(out[:, y0:y1, x0:x1], 1j * torch.ones(1, 6, 6, dtype=torch.complex64), atol=1e-6)


def test_windowed_global_fc_dropout_is_local_to_window():
    fc = GlobalFCPhaseMask(
        canvas_shape=(16, 16),
        phase_size=8,
        phase_mode="center_window",
        phase_init="identity",
        phase_dropout_mode="block_phase_bypass",
        phase_dropout_p=0.5,
        phase_dropout_block_size=4,
    )
    assert fc.phase.phase_dropout_mode == "block_phase_bypass"
    assert fc.phase.phase_dropout_p == 0.5
    fc.train()
    fc.set_phase_dropout_active(True)
    _ = fc(torch.ones(2, 16, 16, dtype=torch.complex64))
    assert fc.phase.last_phase_dropout_mask is not None
    assert fc.phase.last_phase_dropout_mask.shape[-2:] == (8, 8)
