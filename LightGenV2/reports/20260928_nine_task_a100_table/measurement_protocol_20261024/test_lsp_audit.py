"""CPU-only independent metric, crop and strict power support tests."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('lsp_audit', Path(__file__).with_name('audit_lsp.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_official_one_based_crop_and_visibility_channel_ignored():
    joints = np.tile([11., 21., 1.], (14, 1)).astype(np.float32)
    xy, visible, box = m.target_from_original(joints)
    assert box == [-10., 0., 30., 40.]
    assert np.all(xy == 112.) and visible.all()
    joints[:, 2] = 0
    assert np.array_equal(xy, m.target_from_original(joints)[0])


def test_pose_scale_and_missing_torso_not_counted():
    xy = np.zeros((14, 2), dtype=np.float32)
    xy[8] = [10, 0]; xy[9] = [10, 0]; xy[13] = [0, 3]
    mask = np.ones(14, dtype=bool)
    assert m.pose_metric(xy, xy, mask) == (14, 14, 10., 6.)
    mask[[8, 9]] = False
    assert m.pose_metric(xy, xy, mask)[:2] == (0, 0)


def test_power_integrates_whole_window_and_rejects_long_support():
    power = [dict(host_monotonic_s=t, watts=w) for t, w in ((0., 10), (.1, 20), (.2, 30))]
    energy, age = m.board_energy(power, .05, .2)
    assert energy == pytest.approx(2.5) and age == pytest.approx(.1)
    with pytest.raises(ValueError):
        m.board_energy([dict(host_monotonic_s=0, watts=10), dict(host_monotonic_s=.4, watts=10)], 0, .4)
