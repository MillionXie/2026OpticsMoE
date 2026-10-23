import ast,json,hashlib,pytest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]


def test_full300_gate_and_explicit_finite_budget():
    s=(HERE/'lab_caltech20_verified300_G2_20261024.py').read_text();tree=ast.parse(s);n=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='validate');ns={};exec(compile(ast.Module(body=[n],type_ignores=[]),'<validate>','exec'),ns)
    def load(path):
        v=json.loads(path.read_text())
        if 'inherits' in v:
            parent=HERE/v['inherits'];assert hashlib.sha256(parent.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==v['inherits_SHA256'];v={**load(parent),**v}
        return v
    p=load(HERE/'lab_caltech20_verified300_G2_profile_20261024.json');ns['validate'](p)
    for field,value in [('expected_parent_correct',299),('final_fit_seconds',600),('minimum_remaining_seconds',2200)]:
        with pytest.raises(AssertionError):ns['validate']({**p,field:value})
    for name,digest in p['source_SHA256'].items():assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
    assert s.index('can_start_G2(verified,300)')<s.index('fields=capture.acquire(')
    assert "stopped['G2_not_started']" in s and "not (base.ROOT/p['completed_controller'].replace('.return.json','/s_3')).exists()" in s
    assert '==5150' in s and 'base.fit_suffix(' in s and '300' in s and '340' in s and 'automatic_retry=False' in s
