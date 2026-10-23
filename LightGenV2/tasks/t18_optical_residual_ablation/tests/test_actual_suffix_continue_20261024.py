import ast,hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_actual_suffix_continue_20261024.py'


def test_initial_OAS_profile_and_source_identities():
    node=next(n for n in ast.parse(ENTRY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns)
    p=json.loads((ENTRY.parent/'lab_caltech20_actual300_OAS_profile_20261024.json').read_text());ns['validate_policy'](p)
    assert p['method']=='OAS' and p['recipe']['epochs']==1
    for name,digest in p['source_SHA256'].items():assert hashlib.sha256((ENTRY.parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest


def test_old282_keeps_its_own_camera_arrays_and_exact_parent():
    tree=ast.parse(ENTRY.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_policy','load_policy')]
    ns=dict(json=json,hashlib=hashlib,Path=Path,base=SimpleNamespace(HERE=ENTRY.parent));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ENTRY),'exec'),ns)
    for profile in ('lab_caltech20_actual300_FROZEN_profile_20261024.json','lab_caltech20_actual300_RIDGEOLD_profile_20261024.json','lab_caltech20_actual300_SAFE_profile_20261024.json','lab_caltech20_actual300_SOFTOLD_profile_20261024.json','lab_caltech20_actual300_NOISE_profile_20261024.json','lab_caltech20_actual300_METRIC_profile_20261024.json','lab_caltech20_actual300_DESAT_profile_20261024.json','lab_caltech20_actual300_REMAIN_profile_20261024.json','lab_caltech20_actual300_METRIC285_profile_20261024.json','lab_caltech20_actual300_BAND285_profile_20261024.json','lab_caltech20_actual300_PROX285_profile_20261024.json','lab_caltech20_actual300_LOGIT285_profile_20261024.json'):
        p=ns['load_policy'](ENTRY.parent/profile);ns['validate_policy'](p)
        assert p['expected_parent_correct']==(285 if '285' in profile else 282) and p['captures']['2'].endswith('/g1s/s_2/capture')
        for name,digest in p['source_SHA256'].items():
            value=(ENTRY.parent/name).read_bytes().replace(b'\r\n',b'\n')
            if digest=='a65f32b5a750450feddc0d86bfef84cc7660a7583c0bdbd6bcba523fa0608140':value=subprocess.check_output(['git','show','71fef4354:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            if digest=='f82d21638e4dcaefacac2fc4592a3b1b49858595beb4fe50381c999c1439d366':value=subprocess.check_output(['git','show','986f54710:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            if digest=='b4d89807a8f67c99b31b25488642fd60bf9869cccead2c01a2db3206bf860ce6':value=subprocess.check_output(['git','show','98a1d5696:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            if digest=='264ff020ca21c1301201df24b16e61f9fa5448aa3601b6c614659d11cc4927a4':value=subprocess.check_output(['git','show','004053655:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            if digest=='2051709d1fd0bff0dfef5369c2e98eafc4daa82316b0fba02e1b62a4eb7918da':value=subprocess.check_output(['git','show','deb39dc1a:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            if digest=='f77ada65022f9af260f959ac2d592a79ac6371e23fe3a68dcc897947841d03ff':value=subprocess.check_output(['git','show','fb868484a:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            if digest=='07fff8e589129306841eaff0aa4515becfb27a24bc421a4a53d2e55f3443eb4a':value=subprocess.check_output(['git','show','6b6ac6c2c:LightGenV2/tasks/t18_optical_residual_ablation/'+name])
            assert hashlib.sha256(value).hexdigest()==digest


def test_fresh_E2_only_and_immutable_measured_upstream():
    s=ENTRY.read_text();ast.parse(s)
    assert "cap==run/'s_2/capture'" in s and "r['phase_SHA256']==p['E2_phase_SHA256']" in s
    assert "loop.validate_state(m,model,m.state_cpu(capture_model),m.allowed_keys(model,4))" in s
    assert "m.sha(f)==r['arrays'][role]['SHA256']" in s and "metric['correct']==p['expected_parent_correct']" in s
    assert "m.sha(source/'report.json')==sa['report_SHA256']==p['source_report_SHA256']" in s


def test_existing_finite_optimizers_only_and_true_gate_before_G2():
    s=ENTRY.read_text();assert "('OAS','affine_LBFGS','contrast_LBFGS','TRAIN_group_ridge','native_CE','frozen_readout_CE','safe_teacher_CE','supervised_metric_CE','desaturation_CE','local_remaining','bandlimited_CE','parent_proximal')" in s
    assert 'engine.fit_suffix(' in s and 'TRAIN_affine_optimizer=dict(l2_parent_TRAIN_NLL_fraction=.1,label_smoothing=.03,outer_steps=80)' in s
    assert s.index('assert admission.can_start_G2(result,300)')<s.index('fields=capture.acquire(')
    assert "if not admission.can_start_G2(result,300):" in s and 'G2_not_started=True' in s
    assert "==5150" in s and 'loss.backward' not in s
