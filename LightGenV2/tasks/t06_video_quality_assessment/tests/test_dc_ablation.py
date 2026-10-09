"""Physics and opt-in guard for the DC intervention; no real data or devices."""
import dataclasses
import json
from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import load_settings
from LightGenV2.tasks.t06_video_quality_assessment.dc_ablation import ablation_settings, scratch_settings, state_fingerprint
from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import resolved_dict
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


def test_scratch_uses_json_only_and_fixed_rho(tmp_path, monkeypatch):
    import LightGenV2.tasks.t06_video_quality_assessment.dc_ablation as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Scratch must not read a student checkpoint')
    monkeypatch.setattr(module, 'load_model', forbidden)
    template=tmp_path/'architecture.json'
    template.write_text(json.dumps(resolved_dict(settings())))
    yes=scratch_settings(template,Path('scratch_dc_ablation_dc20'),'dc20')
    no=scratch_settings(template,Path('scratch_dc_ablation_dc0'),'dc0')
    assert yes.initialization_checkpoint is None and no.initialization_checkpoint is None
    assert yes.geometry==no.geometry and yes.top_k==no.top_k==2
    assert (yes.unmodulated_power_fraction_min,yes.unmodulated_power_fraction_max,yes.unmodulated_power_fraction_eval)==(.2,.2,.2)
    assert (no.unmodulated_power_fraction_min,no.unmodulated_power_fraction_max,no.unmodulated_power_fraction_eval)==(0.,0.,0.)
    assert yes.phase_snapshot_interval_epochs==no.phase_snapshot_interval_epochs==0


def test_initial_state_fingerprint_records_actual_parameters():
    torch.manual_seed(163);one=torch.nn.Linear(4,3)
    torch.manual_seed(163);two=torch.nn.Linear(4,3)
    assert state_fingerprint(one)==state_fingerprint(two)
    with torch.no_grad():two.weight[0,0]+=1
    assert state_fingerprint(one)!=state_fingerprint(two)
