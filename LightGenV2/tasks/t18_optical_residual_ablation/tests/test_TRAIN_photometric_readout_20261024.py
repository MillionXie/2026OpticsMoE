import ast,hashlib,json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1];ENTRY=HERE/'lab_caltech20_TRAIN_photometric_readout_20261024.py'


def policy():return json.loads((HERE/'lab_caltech20_TRAIN_photometric_readout_profile_20261024.json').read_text())


def validate(p):
    node=next(v for v in ast.parse(ENTRY.read_text()).body if isinstance(v,ast.FunctionDef) and v.name=='validate_policy')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns);ns['validate_policy'](p)


def test_exact_new_mask_source_and_finite_budget():
    p=policy();validate(p)
    for key,name in [('mask_engine_SHA256','lab_caltech20_TRAIN_photometric_mask_20261024.py'),('linear_engine_SHA256','lab_caltech20_tucam_TRAIN_linear_20261023.py'),('mask_policy_SHA256',p['mask_policy'])]:assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]
    for k,v in [('mask_correct',273),('mask_TRAIN_correct',1600),('TEST_gradient',True),('new_inference_layers',1),('intermediate_target',258),('SDK_used',True),('GPU_used',True),('scheduler',True),('VALID',411)]:
        with pytest.raises(AssertionError):validate({**p,k:v})
    with pytest.raises(AssertionError):validate({**p,'recipe':{**p['recipe'],'outer_steps':200}})


def test_fresh_complete_cache_and_parent_probability_identity():
    s=ENTRY.read_text()
    assert "(('train',1600),('test',411))" in s and "errors[role]<=3e-6" in s
    assert "source/('best_'+role+'_predictions.npz')" in s and "np.array_equal(z['ids'],data[role][2])" in s
    assert "keys={'linear.weight','linear.bias'}" in s and 'loop.validate_state(m,native,original,keys)' in s
    assert "list(fields)==['router','0','1']" in s and "local_SHA_backup.json" in s


def test_TRAIN_only_objective_and_single_initialization_fit():
    s=ENTRY.read_text();body=s.split('def closure():',1)[1].split('opt.step(closure)',1)[0]
    assert 'TRAIN_affine_objective(head.weight,head.bias,train,labels,weights' in body and "data['test']" not in body
    assert "labels=torch.from_numpy(data['train'][1]" in s and 'TRAIN_linear_fit(train.numpy(),labels.numpy(),prior)' in s
    assert 'use_analytical=analytical_objective<initial_objective' in s and 'initialization_chosen_by_TRAIN_only=True' in s
    assert "cfg['L2_parent_TRAIN_NLL_fraction']*source_report['results']['best']['calibrated_train']['nll']" in s


def test_no_camera_or_physical_accuracy_claim_and_full_CPU_replay():
    s=ENTRY.read_text()
    assert 'scorer(native,gain,bg,data,fields,role)' in s and 'selection_not_physical_accuracy=True' in s
    assert 'actual_source_correct_still273=True,new_CCD=0,G2_not_started=True' in s
    assert 'next_requires_original_physical_E2_gate305=True' in s
    for bad in ('capture.acquire(','Bench(','torch.nn.Sequential','torch.cuda','TEST_mixup','scheduler.run'):assert bad not in s
