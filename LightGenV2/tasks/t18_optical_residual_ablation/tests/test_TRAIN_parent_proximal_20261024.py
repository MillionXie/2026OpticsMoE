import ast
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ENTRY=HERE/'lab_caltech20_TRAIN_parent_proximal_20261024.py'


def test_TRAIN_objective_has_no_development_arguments():
    s=ENTRY.read_text();tree=ast.parse(s);n=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='objective');body=ast.get_source_segment(s,n)
    assert 'development' not in body and 'test' not in body.lower() and 'teacher.detach()' in body
    assert "target.argmax(1)==labels" in body and 'weight-origin' in body


def test_native_fold_identity_and_bounded_original_row_delta():
    s=ENTRY.read_text();assert "raw_parent[k]+(v-origin[k])" in s and '.075/norm' in s
    assert 'radius<=.080001' in s and 'zero_delta_native_identity_exact=True' in s and 'new_inference_layers=0' in s
    assert 'range(1,801)' in s and 'TRAIN_consumed=1600' in s and 'with torch.inference_mode()' in s


def test_head_engine_keeps_source_fallback_and_fresh_native_cache():
    s=(HERE/'lab_caltech20_actual_readout_20261024.py').read_text()
    assert "TRAIN_parent_proximal=True" in s and "best_head=raw_parent" in s
    assert "cache=out/'native_TRAIN_features.npz'" in s and "for begin in range(0,1600,8):" in s
    assert "correction['last_head']if proximal" in s and 'all_non_readout_frozen=True' in s
