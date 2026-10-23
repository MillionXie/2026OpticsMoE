import ast,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parents[1];ENTRY=HERE/'lab_caltech20_TRAIN_native_mask_20261024.py'


def test_pinned_response_and_finite_recipe():
    s=ENTRY.read_text();tree=ast.parse(s);node=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='pinned_response_sha');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns)
    assert hashlib.sha256((HERE/'lab_caltech20_TRAIN_native_response_fit_20261024.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==ns['pinned_response_sha']()
    assert 'epochs=24,seconds=1200' in s and 'range(1,25)' in s and 'time.monotonic()-start>=1200' in s
    assert "rr['native_response_admitted']" in s and "rr['actual_updates']==200" in s and "m.read(folder.with_suffix('.return.json'))['returncode']==0" in s


def test_original_state_wrapper_and_measured_callback_bypass():
    s=ENTRY.read_text().split('def make_predictor(',1)[1].split('def score(',1)[0]
    assert "stage in self.measured or callback is not None" in s and 'return super().capture(' in s
    assert 'self.load_state_dict(m.state_cpu(model),strict=True)' in s and 'assert set(work.state_dict())==set(model.state_dict())' in s
    assert 'response(full,full.new_tensor(self._gain)' in s
    for bad in ('nn.Parameter','register_buffer','torch.save','grid_sample'):assert bad not in s


def test_TEST_only_development_no_gradients_and_old_E2_not_reused():
    s=ENTRY.read_text();body=s.split('for begin in range(0,1600,8):',1)[1].split('assert used==1600',1)[0]
    assert "data['train'][1][ix]" in body and "data['test']" not in body
    assert "list(fields)==['router','0','1']" in s and 'parent.admit(p,c,a.out)' in s
    assert 'selection_not_physical_accuracy=True' in s and 'cpu=make_predictor(model,gain,bg,sigma).eval()' in s
    assert "metric,prob=score(native,g,b,sigma,data,fields,role)" in s


def test_original_actual_gate_enforced_before_G2_and_fixed_hardware():
    s=ENTRY.read_text().split('def run(a):',1)[1]
    assert "if stage=='2' and result['selected_correct']<305:" in s
    assert "if stage=='3':assert result['strict_original_CPU'] and result['results']['best']['test']['correct']>=305" in s
    assert s.index("if stage=='3':assert result['strict_original_CPU']")<s.index('fields=capture.acquire(')
    assert "new_phase!=old_phase" in s and 'bool(changed)' in s and "not rr['selected_E0']" in s
    assert "m.sha(hardware)==c['hardware_by_stage'][stage]['SHA256']" in s
    assert 'response_map_not_deployed=True' in s and 'prefix_SHA256=base.identities(m,paths)' in s
