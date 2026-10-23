import ast,hashlib,json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1];ENTRY=HERE/'lab_caltech20_TRAIN_photometric_mask_20261024.py'


def policy():return json.loads((HERE/'lab_caltech20_TRAIN_photometric_mask_profile_20261024.json').read_text())


def validate(p):
    node=next(v for v in ast.parse(ENTRY.read_text()).body if isinstance(v,ast.FunctionDef) and v.name=='validate_policy')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns);ns['validate_policy'](p)


def test_published_source_and_admitted_response_fixed_original_best():
    p=policy();validate(p)
    for key,name in [('base_profile_SHA256',p['base_profile']),('suffix_engine_SHA256','lab_caltech20_tucam_layerwise_20261023.py'),('capture_engine_SHA256','lab_caltech20_tucam_capture_20261023.py'),('operator_SHA256','lab_caltech20_TRAIN_response_operator_20261024.py'),('flat_engine_SHA256','lab_caltech20_TRAIN_flat_response_20261024.py'),('phase_probe_engine_SHA256','lab_caltech20_tucam_E2_mask_retake_20261023.py')]:
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]
    for k,v in [('source_correct',282),('TEST_gradient',True),('intermediate_target',273),('new_inference_layers',1),('scheduler',True),('new_G2_if_gate_unmet',True)]:
        with pytest.raises(AssertionError):validate({**p,k:v})


def test_old_E2_only_source_baseline_then_remove_from_every_mask_training_field():
    s=ENTRY.read_text();a=s.split('def admit(',1)[1].split('def fit_mask(',1)[0]
    assert "score['correct']==273" in a and "fields.pop('2');paths.pop('2')" in a
    assert "m.allowed_keys(model,4)" in a and 'old_E2_not_used_after_mask_changes=True' in a
    assert "r['response_admitted']" in a and "m.sha(response/'TRAIN_flat_gain_background.npz')" in a
    assert 'old_E2_used_only_for_source_baseline=True' in a


def test_predictions_are_not_physical_and_CPU_subclass_not_lost_via_GPU_assess():
    s=ENTRY.read_text();body=s.split('def predictive_score(',1)[1].split('def admit(',1)[0]
    assert 'cpu=make_predictor(model,gain,bg).eval()' in body and 'base.evaluate(cpu' in body
    assert 'calibrated_hybrid_prediction_NOT_physical_accuracy' in s
    assert "results[label]['nominal_unmeasured_TEST_counterfactual']" in s and "results['best']['calibrated_test']['correct']==best" in s
    assert "loop.put_fields(work,data,fields,'train'" in s and "labels=torch.from_numpy(data['train']" in s
    assert "data['test']" not in s.split('for begin in range(0,1600,8):',1)[1].split('assert used==1600',1)[0]


def test_actual_original_CPU_E2_gate_before_G2_capture_and_last_5150_only():
    s=ENTRY.read_text().split('def run(a):',1)[1]
    assert "new_phase!=old_phase" in s and "not fit['selected_E0']" in s and 'bool(changed)' in s
    assert "fields=capture.acquire(model" in s and "m.read(paths['2']/'report.json')['phase_SHA256']==new_phase" in s
    assert s.index("if result['selected_correct']<305:")<s.index("folder=a.out/'s_3'")
    assert "result['results']['best']['test']['correct']>=305" in s
    assert "m.allowed_keys(model,5))==5150" in s and "full_physical=True" in s
    assert "shutil.disk_usage(base.ROOT).free>=120*1024**3" in s


def test_photometric_operator_has_native_state_and_measured_bypass():
    s=(HERE/'lab_caltech20_TRAIN_response_operator_20261024.py').read_text().split('class TRAINPhotometricPrefix',1)[1].split('def selftest(',1)[0]
    assert 'self.load_state_dict(state,strict=True)' in s and "stage in self.measured" in s
    assert 'self.physical_fields(amplitude,stage,weights)' in s and 'self.propagator(field).abs().square()' in s and 'apply_map(nominal' in s
    assert 'nominal_original_probability_maxabs' in s and 'phase_gradient_norms=norms' in s and 'real_CCD_bypass_exact=True' in s
    for bad in ('nn.Parameter','register_buffer','torch.save','grid_sample','Bench('):assert bad not in s
