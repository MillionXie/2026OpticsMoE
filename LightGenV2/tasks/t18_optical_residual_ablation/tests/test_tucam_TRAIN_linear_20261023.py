import ast,hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_TRAIN_linear_20261023.py'


def test_TRAIN_numerical_estimator_and_affine_invariance():
    tree=ast.parse(ENTRY.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('TRAIN_linear_fit','selftest')]
    ns={'json':__import__('json')};exec(compile(ast.Module(body=functions,type_ignores=[]),str(ENTRY),'exec'),ns);ns['selftest']()


def test_no_inference_layer_or_TEST_estimation():
    s=ENTRY.read_text();tree=ast.parse(s);fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_linear_fit')
    assert 'test' not in [n.id for n in ast.walk(fit) if isinstance(n,ast.Name)]
    assert "keys={'linear.weight','linear.bias'}" in s and 'all_non_readout_frozen=True' in s
    assert "features,data['train'][1],prior" in s and "put_fields(model,data,fields,'train'" in s
    assert 'loop.validate_state(m,native,original,keys)' in s and 'new_inference_layers=0' in s
    assert 'selected_epoch=None' in s and 'analytical_fits=1' in s
    assert 'hyperparameter_grid=False' in s and 'GPU_used=False' in s
    assert '2<=len(fields)<=4 and list(fields)==list(base.STAGES[:len(fields)])' in s


def test_exact_engine_and_linear_only_policy():
    p=json.loads((HERE/'lab_caltech20_tucam_E1_linear_profile_20261023.json').read_text())
    assert hashlib.sha256(ENTRY.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    driver=HERE/'lab_caltech20_tucam_E1_underfit_20261023.py';tree=ast.parse(driver.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(driver),'exec'),ns);ns['validate_policy'](p)
    assert p['minimum_parent_TRAIN_TEST_gap']==.1 and p['learning_rates']=={}
    s=driver.read_text();assert s.index('linear.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')


def test_cached_TRAIN_augmentation_policy_and_identity():
    p=json.loads((HERE/'lab_caltech20_tucam_E1_feature_aug_profile_20261023.json').read_text())
    assert hashlib.sha256(ENTRY.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    driver=HERE/'lab_caltech20_tucam_E1_underfit_20261023.py';tree=ast.parse(driver.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(driver),'exec'),ns);ns['validate_policy'](p)
    s=ENTRY.read_text();assert 'max(error,train_error)<=3e-6' in s and "sp['prefix_SHA256']==lineage" in s
    assert 'TRAIN_mixup(train[ix],labels[ix]' in s and 'TRAIN_mixup(test' not in s
    assert 'head(test)' in s and 'with torch.inference_mode():' in s
    assert "loop.validate_state(m,native,original,keys)" in s and "results['best']['test']['correct']==best" in s


def test_best304_cooldown_and_head_independent_cache_reuse():
    p=json.loads((HERE/'lab_caltech20_tucam_E1_feature_cooldown_profile_20261023.json').read_text())
    assert hashlib.sha256(ENTRY.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    assert p['expected_parent_correct']==304 and p['TRAIN_feature_aug']['cooldown'] is True
    s=ENTRY.read_text();assert '1-strength*(1-rng.beta' in s and "old.linear(train)" in s
    assert "sr['results']['best']['PT_SHA256']==m.sha(source/'best.pt')" in s
    assert 'loop.validate_state(m,old,original,keys)' in s and "cfg['dropout']*strength" in s


def test_fullbatch_objective_and_L2_are_TRAIN_only_and_native_folded():
    s=ENTRY.read_text();tree=ast.parse(s);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_affine_objective')
    assert 'test' not in {v.id for v in ast.walk(f) if isinstance(v,ast.Name)}
    assert "parent_report['results']['best']['train']['nll']" in s
    assert 'regularization=2*cfg[\'l2_parent_TRAIN_NLL_fraction\']*train_nll/norm' in s
    assert 'TRAIN_affine_objective(head.weight,head.bias,train64,labels,weights' in s
    assert 'F.linear(test,head.weight.float(),head.bias.float())' in s
    assert 'full_TRAIN_gradient_evaluations=closure_calls' in s


def test_fresh_TRAIN_cache_stays_in_new_run_and_preserves_parent():
    s=ENTRY.read_text();assert "if c.get('fresh_native_TRAIN_features',False):" in s
    assert "cache=out/'native_TRAIN_features.npz'" in s
    assert "method='fresh_native_parent_TRAIN_cache_only'" in s
    assert "source_PT_SHA256=sr['results']['best']['PT_SHA256']" in s
    assert 'loop.validate_state(m,old,original,keys)' in s and 'max(error,train_error)<=3e-6' in s


def test_local_boundary_numerics_and_TRAIN_only_gradient(monkeypatch):
    import importlib.util
    p=HERE/'lab_caltech20_TRAIN_local_boundary_20261023.py'
    spec=importlib.util.spec_from_file_location('local_boundary_check',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.selftest()
    tree=ast.parse(p.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_direction')
    names={n.id.lower() for n in ast.walk(f) if isinstance(n,ast.Name)}
    assert not names.intersection({'test','development','development_labels','dy','z'})
    s=p.read_text();assert 'np.array_equal(ww[fixed],parentw[fixed])' in s and "gradient_data='TRAIN_only'" in s
    assert 'exec(' not in s and 'FunctionType' not in s


def test_local_profile_native_fresh_cache_and_fixed_gate(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    p=json.loads((HERE/'lab_caltech20_tucam_G1_local_boundary_profile_20261023.json').read_text())
    assert hashlib.sha256(ENTRY.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    kernel=HERE/'lab_caltech20_TRAIN_local_boundary_20261023.py'
    # This closed profile pins its ORIGINAL released source. The current kernel
    # only adds another audited finite class plan; all numerical bodies are exact.
    legacy=subprocess.check_output(['git','show','849c502dc:LightGenV2/tasks/t18_optical_residual_ablation/lab_caltech20_TRAIN_local_boundary_20261023.py'],cwd=HERE.parents[2])
    assert hashlib.sha256(legacy).hexdigest()==p['TRAIN_local_engine_SHA256']
    old={f.name:ast.dump(f,include_attributes=False) for f in ast.parse(legacy.decode()).body if isinstance(f,ast.FunctionDef) and f.name!='validate'}
    current={f.name:ast.dump(f,include_attributes=False) for f in ast.parse(kernel.read_text()).body if isinstance(f,ast.FunctionDef) and f.name!='validate'}
    assert old==current
    driver=HERE/'lab_caltech20_tucam_suffix_response_20261023.py';tree=ast.parse(driver.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(driver),'exec'),ns);ns['validate_policy'](p)
    assert p['fresh_native_TRAIN_features'] is True and p['expected_parent_correct']==301 and p['intermediate_target']==305
    assert "per_update_parameters=257" in ENTRY.read_text() and "all_non_readout_frozen=True" in ENTRY.read_text()


def test_remaining_boundary_is_one_finite_coordinate_pass(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    p=json.loads((HERE/'lab_caltech20_tucam_G1_remaining_boundary_profile_20261023.json').read_text())
    driver=HERE/'lab_caltech20_tucam_suffix_response_20261023.py';tree=ast.parse(driver.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(driver),'exec'),ns);ns['validate_policy'](p)
    assert p['TRAIN_local_boundary']['maximum_classes']+len(p['TRAIN_local_boundary']['skip_classes'])==20
    assert p['expected_parent_correct']==304 and p['fresh_native_TRAIN_features'] is False
    s=ENTRY.read_text();assert "sr['local_plan_SHA256']==m.sha(plan)" in s and "m.read(plan)]==cfg['skip_classes']" in s
