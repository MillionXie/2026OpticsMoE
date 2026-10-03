import math
import torch
from LightGenV2.tasks.t02_keypoint_detection.global_noise import uniform_phase_raw


def test_noise_is_fixed_reproducible_uniform_physical_phase():
    raw=uniform_phase_raw((478,478),42)
    assert torch.equal(raw,uniform_phase_raw((478,478),42))
    assert not torch.equal(raw,uniform_phase_raw((478,478),43))
    phase=2*math.pi*raw.sigmoid()
    assert phase.min()>=0 and phase.max()<=2*math.pi
    assert abs(float(phase.mean())-math.pi)<.02
    assert not raw.requires_grad
