import ast,hashlib,json
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


def test_exact_engine_and_linear_only_policy():
    p=json.loads((HERE/'lab_caltech20_tucam_E1_linear_profile_20261023.json').read_text())
    assert hashlib.sha256(ENTRY.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    driver=HERE/'lab_caltech20_tucam_E1_underfit_20261023.py';tree=ast.parse(driver.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(driver),'exec'),ns);ns['validate_policy'](p)
    assert p['minimum_parent_TRAIN_TEST_gap']==.1 and p['learning_rates']=={}
    s=driver.read_text();assert s.index('linear.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')
