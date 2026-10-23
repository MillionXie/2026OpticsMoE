import ast,hashlib,json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_TRAIN_response_20261024.py'


def functions():
    tree=ast.parse(ENTRY.read_text())
    picked=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ('validate_policy','validate_contrast','validate_spatial','validate_polish','anchors')]
    ns={};exec(compile(ast.Module(body=picked,type_ignores=[]),str(ENTRY),'exec'),ns);return ns


def test_profile_published_numerical_engines_and_new_TUCam_identity():
    p=json.loads((HERE/'lab_caltech20_tucam_TRAIN_response_profile_20261024.json').read_text());functions()['validate_policy'](p)
    for key,name in [('base_profile_SHA256',p['base_profile']),('suffix_engine_SHA256','lab_caltech20_tucam_layerwise_20261023.py'),('response_engine_SHA256','lab_caltech20_dvp_TRAIN_response_fit_20261020.py'),('patterns_engine_SHA256','lab_caltech20_dvp_TRAIN_response_20261020.py')]:
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]
    for k,v in [('TEST_used',True),('LUT_modified',True),('intermediate_target',273),('original_TRAIN_per_class',3),('hardware_SHA256','0'*64)]:
        with pytest.raises(AssertionError):functions()['validate_policy']({**p,k:v})


def test_original_TRAIN_balanced_split_never_augmented_TEST_or_VALID():
    rows=[]
    for i in range(3):
        rows.extend(dict(label=k,kind='augmented' if i==0 else 'original',sample_id=f'{i}_{k}') for k in range(20))
    a=functions()['anchors'](rows)
    assert len(a)==40 and len({v['index'] for v in a})==40
    assert [v['split'] for v in a]==['fit','held_TRAIN']*20
    assert min(v['index'] for v in a)>=20
    with pytest.raises(AssertionError):functions()['anchors'](rows[:40])


def test_only_TRAIN_canonical_no_dataset_or_prediction_loading():
    s=ENTRY.read_text()
    assert "cap/'train'" in s and "cap/'test'" not in s
    assert 'data_for(' not in s and "['test']" not in s and 'linear.' not in s
    assert "receipt['canonical_SHA256']" in s and "receipt['sample_id']==row['sample_id']" in s
    assert "r['arrays']['train']['SHA256']" in s and "np.array_equal(m.phase_native(phase)" in s


def test_current_TUCam_original_encoding_guard_and_nominal_restore_finally():
    s=ENTRY.read_text();assert 'from lab_caltech20_tucam_session_20261022 import Bench,logical' in s
    assert 'from lab_caltech20_tucam_L6_20261022 import check_connection' in s
    assert "m.native(drive[gray],'amplitude')" in s and 'm.phase_native(pixels)' in s
    assert "row['p99']>=15 and row['saturation_fraction']<=.15" in s
    assert 'finally:\n                bench.stage(nominal);restored=True' in s
    assert "protected=={q:m.sha(q) for q in protected}" in s
    assert "not m.LOCK.exists()" in s and 'automatic_retry=False' in s
    assert 'set_exposure(' not in s and 'calibrate(' not in s


def test_response_holdout_nonflat_admission_not_classification_or_deployed_model():
    s=ENTRY.read_text().split('def fit(',1)[1].split('def selftest(',1)[0]
    assert "len(train)==len(held)==84" in s and "range(1,161)" in s
    assert "final['held_TRAIN']['loss']<=baseline['held_TRAIN']['loss']*.95" in s
    assert "final['held_TRAIN']['coarse_PCC']>=baseline['held_TRAIN']['coarse_PCC']+.02" in s
    assert "stable,'Within-connection TRAIN repeats unstable" in s
    assert "raw=torch.nn.Parameter" in s and 'cross_entropy' not in s and 'saved_model(' not in s
    assert 'physical_LUT_calibrated=False' in s and 'new_inference_layers=0' in s
    assert 'logical(raw,warp,65535' in s and "m.sha(row['raw'])==row['raw_SHA256']" in s


def test_rejected_brightness_fit_only_allows_fixed_four_parameter_contrast_comparison():
    p=json.loads((HERE/'lab_caltech20_tucam_TRAIN_response_contrast_profile_20261024.json').read_text());functions()['validate_contrast'](p)
    assert hashlib.sha256((HERE/'lab_caltech20_dvp_TRAIN_spatial_response_20261020.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['spatial_engine_SHA256']
    s=ENTRY.read_text();assert "objective=spatial.spatial_loss" in s
    assert "not prior['response_fit_admitted']" in s and "prior['probe_report_SHA256']==m.sha(probes/'report.json')" in s
    assert "final['held_TRAIN']['coarse_PCC']>=start_metrics['held_TRAIN']['coarse_PCC']+.02" in s
    four=s.split('def fit(',1)[1].split('def selftest(',1)[0]
    assert 'wavefront_basis(' not in four and 'monotonic_curve(' not in four
    assert 'if spatial_policy is not None:' in four
    for key,value in [('shared_parameters',27),('new_CCD',208),('TEST_used',True),('fit_steps',400)]:
        with pytest.raises(AssertionError):functions()['validate_contrast']({**p,key:value})


def test_spatial_response_uses_shared_TRAIN_nuisance_not_deployed_layer_and_normal_exit():
    p=json.loads((HERE/'lab_caltech20_tucam_TRAIN_response_spatial_profile_20261024.json').read_text());functions()['validate_spatial'](p)
    s=ENTRY.read_text().split('def fit_spatial(',1)[1].split("if __name__",1)[0]
    assert 'spatial.monotonic_curve' in s and 'spatial.spatial_intensity' in s
    assert 'sum(v.numel() for v in parameters)==27' in s and 'range(1,401)' in s
    assert "f['PCC_by_condition']['nominal_before']>=b['PCC_by_condition']['nominal_before']" in s
    assert 'labels_used=False' in s and 'model_checkpoint_written=False' in s
    assert 'torch.save(' not in s and 'saved_model(' not in s and 'Bench(' not in s
    assert 'raise SystemExit(0)' not in ENTRY.read_text()
    for key,value in [('new_inference_layers',1),('image_warp',True),('held_PCC_gain',0),('nominal_held_PCC_not_lower',False)]:
        with pytest.raises(AssertionError):functions()['validate_spatial']({**p,key:value})


def test_fullbatch_finish_keeps_spatial_gate_and_gradients_only_fit_TRAIN():
    p=json.loads((HERE/'lab_caltech20_tucam_TRAIN_response_polish_profile_20261024.json').read_text());functions()['validate_polish'](p)
    s=ENTRY.read_text();closure=s.split('def closure():',1)[1].split('opt.step(closure)',1)[0]
    assert 'range(0,84,4)' in closure and 'for v in tr' in closure
    assert 'held' not in closure and 'labels' not in closure and "'test'" not in closure
    assert 'max_iter=12,max_eval=24' in s and "prev['state']=='complete' and not prev['spatial_response_admitted']" in s
    assert "final['TRAIN']['loss']<initial['TRAIN']['loss']" in s
    for k,v in [('gate_unchanged',False),('max_iter',100),('shared_parameters',28)]:
        with pytest.raises(AssertionError):functions()['validate_polish']({**p,k:v})
