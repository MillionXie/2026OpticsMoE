import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.optics.expert_layout import ExpertLayout


def test_fast120_default_active_window_covers_expert_union():
    layout = ExpertLayout()
    layout.validate()
    assert layout.geometry_profile == "fast120_520"
    assert layout.expert_union_bounds == [50, 470, 50, 470]
    assert layout.expert_union_size == 420
    assert layout.active_window_size == 450
    assert [layout.active_window_aperture.y0, layout.active_window_aperture.y1] == [35, 485]
    assert layout.active_window_size >= layout.expert_union_size


def test_explicit_fair134_layout_remains_compatible():
    layout = ExpertLayout(
        num_experts=9,
        canvas_size=1000,
        input_size=134,
        expert_size=134,
        expert_pitch=200,
        padding=200,
        prompt_aperture_size=600,
    )
    layout.validate()
    assert layout.expert_union_bounds == [233, 767, 233, 767]
    assert layout.expert_union_size == 534
    assert layout.active_window_size == 600
    assert layout.active_window_aperture.y0 == 200
    assert layout.active_window_aperture.y1 == 800
    assert layout.active_window_aperture.x0 == 200
    assert layout.active_window_aperture.x1 == 800
    assert layout.active_window_size >= layout.expert_union_size
    assert layout.to_dict()["active_window_aperture"]["center"] == [500, 500]
    assert layout.geometry_profile == "fair134_1000"
