import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccd import sample_camera
from build_projects import build, PROJECTS


def test_camera_statistics_and_gradient():
    torch.manual_seed(71)
    clean = torch.full((250000,), 2.0, requires_grad=True)
    profile = {"electrons_per_intensity_unit": 100, "read_noise_electrons": 5, "dark_electrons": 3}
    noisy = sample_camera(clean, profile)
    # Away from clipping: Var(Y)=(k*I+d+sigma**2)/k**2.
    assert abs(float(noisy.detach().mean()) - 2) < 0.003
    assert abs(float(noisy.detach().var()) - 0.0228) < 0.0005
    noisy.sum().backward()
    assert torch.equal(clean.grad, torch.ones_like(clean))
    assert sample_camera(clean, profile, scale=0) is clean
    with pytest.raises(ValueError):
        sample_camera(clean, profile, scale=float("nan"))


def test_four_projects_are_portable_and_locked(tmp_path):
    output = build(tmp_path / "four")
    for group, directory in PROJECTS.items():
        project = output / directory
        assert json.loads((project / "project.json").read_text())["group"] == group
        assert not (project / "weights/best_checkpoint.pt").exists()
        result = subprocess.run([sys.executable, "-I", str(project / "project.py"), "plan"],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([sys.executable, "-I", str(project / "project.py"), "plan", "--group", "r0_post"],
                                capture_output=True, text=True)
        assert result.returncode != 0
    with pytest.raises(ValueError):
        build(output)


def test_camera_patch_restored():
    sys.path.insert(0, str(ROOT / "runtime"))
    from LightGenV2.tasks.t06_video_quality_assessment.models import multivideo9x4 as optics
    from ccd import camera_operator
    from types import SimpleNamespace
    original = optics._perturb_ccd
    profile = {"electrons_per_intensity_unit": 100, "read_noise_electrons": 2, "dark_electrons": 0}
    clean = torch.ones((8, 8))
    with camera_operator(profile):
        assert optics._perturb_ccd(clean, SimpleNamespace(ccd_noise_enabled=False), training=True) is clean
        assert optics._perturb_ccd(clean, SimpleNamespace(ccd_noise_enabled=True), training=False) is clean
        assert not torch.equal(optics._perturb_ccd(clean, SimpleNamespace(ccd_noise_enabled=True), training=True), clean)
    assert optics._perturb_ccd is original
