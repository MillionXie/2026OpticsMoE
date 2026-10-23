import ast
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_TRAIN_response_operator_20261024.py'


def test_original_state_schema_and_no_deployment_module_or_patch():
    s=ENTRY.read_text();ast.parse(s)
    assert 'class TRAINResponsePrefix(MeasuredPrefix)' in s and 'self.load_state_dict(state,strict=True)' in s
    assert 'set(self.state_dict())==set(state)' in s
    for bad in ('register_buffer','nn.Parameter','torch.save','Bench(','.__class__=','MethodType','set_exposure','coordinate'):
        assert bad not in s


def test_original_quantized_amplitude_phase_STE_and_measured_bypass():
    s=ENTRY.read_text();body=s.split('def capture(',1)[1].split('def selftest(',1)[0]
    assert "stage!='2' or stage in self.measured or callback is not None" in body
    assert 'self.physical_fields(amplitude,stage,weights)' in body and 'turns=turns+(exact-turns).detach()' in body
    assert 'spatial_intensity(amp,turns,self.propagator' in body
    assert 'E2_phase_gradient_norms=phase_norms' in s and 'original_model_reload_exact=True' in s
    assert 'measured_CCD_bypass_exact=True' in s and 'new_inference_layers=0' in s
