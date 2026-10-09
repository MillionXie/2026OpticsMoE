"""Physics and opt-in guard for the DC intervention; no real data or devices."""
import dataclasses
from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import load_settings
from LightGenV2.tasks.t06_video_quality_assessment.dc_ablation import ablation_settings
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling import _phase_modulation


def settings():
    return load_settings(Path(__file__).parents[1]/'configs/lightgen/temporal_multivideo16x4_formal.yaml')


def test_dc_zero_requires_explicit_ablation_and_named_run():
    base=settings()
    with pytest.raises(ValueError,match='20%'):
        dataclasses.replace(base,unmodulated_power_fraction_min=0.,unmodulated_power_fraction_max=0.,unmodulated_power_fraction_eval=0.).validate()
    with pytest.raises(ValueError,match='dc_ablation'):
        ablation_settings(base,Path('ordinary_formal_run'),'dc0')
    changed=ablation_settings(base,Path('temporal_dc_ablation_dc0'),'dc0')
    assert changed.geometry==base.geometry and changed.top_k==2
    assert changed.phase_snapshot_interval_epochs==0


def test_complex_field_formula_and_power_is_not_final_ccd_fraction():
    base=settings()
    raw=torch.tensor([[-1.,0.,1.]],requires_grad=True)
    phase=2*torch.pi*torch.sigmoid(raw)
    zero=ablation_settings(base,Path('temporal_dc_ablation_dc0'),'dc0')
    dc0=_phase_modulation(raw,settings=zero,training=False)
    dc20=_phase_modulation(raw,settings=base,training=False)
    torch.testing.assert_close(dc0,torch.exp(1j*phase))
    torch.testing.assert_close(dc20,(.8**.5)*torch.exp(1j*phase)+(.2**.5))
    assert not torch.allclose(dc20.abs().square(),torch.ones_like(raw))
    dc0.real.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
