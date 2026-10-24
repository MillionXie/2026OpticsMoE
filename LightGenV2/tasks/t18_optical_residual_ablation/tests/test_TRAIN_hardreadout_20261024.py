import ast
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]


def test_gradient_objective_has_no_development_inputs():
    tree=ast.parse((HERE/'lab_caltech20_TRAIN_hardreadout_20261024.py').read_text())
    fun=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='objective')
    assert [v.arg for v in fun.args.args]==['logits','labels','importance','parent_prob','delta','bias_delta']
    assert not any(isinstance(v,ast.Name)and v.id in ('development','development_labels','zy','z')for v in ast.walk(fun))


def test_optimizer_only_folds_original_head():
    src=(HERE/'lab_caltech20_TRAIN_hardreadout_20261024.py').read_text()
    assert 'delta@precondition' in src and "'new_inference_layers':" not in src
    assert "new_inference_layers=0" in src and 'canonical_affine_logits' in src
    assert 'TRAIN_consumed=1600' in src and 'time.monotonic()-started>=c' in src
    assert 'model.' not in src and 'cuda' not in src.lower()


def test_parent_and_lineage_admission_preserved():
    src=(HERE/'lab_caltech20_fullG2_OAS_20261024.py').read_text()
    assert '105a0f517cd8dda1a4e5bb7f991505476ead763784e828a663921995f7cbf1a6' in src
    assert 'ca[\'matched_samples\']==2011' in src and "loop.validate_state" in src
    ast.parse(src)
