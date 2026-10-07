"""T12 legacy dependency boundary; AST only, never import model or devices."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T07 = ROOT / 'LightGenV2/tasks/t07_abo_image_retrieval'
T12 = ROOT / 'LightGenV2/tasks/t12_text_to_image/lab_shs8um'


def tree(path):
    return ast.parse(path.read_text(encoding='utf8'))


def functions(node):
    return {n.name: ast.dump(n, include_attributes=False) for n in node.body
            if isinstance(n, ast.FunctionDef)}


def test_t12_geometry_calls_match_preserved_legacy_helpers():
    source = tree(T12 / 'run_layerwise.py')
    used = {n.func.attr for n in ast.walk(source) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.value.id == 'flow'}
    assert used == {'sha', 'phase_gray'}
    bounded = tree(T12 / 'bounded_amplitude.py')
    assert any(isinstance(n, ast.Attribute) and n.attr == 'active_to_native'
               for n in ast.walk(bounded))
    old = functions(tree(T07 / 'lab_dvp8um/four_image_flow.py'))
    new = functions(tree(T07 / 'hardware/geometry.py'))
    for name in ('sha', 'phase_gray', 'active_to_native'):
        assert old[name] == new[name]


def test_capture_lifecycle_unchanged_but_constructor_not_interchangeable():
    def bench(path):
        return next(n for n in tree(path).body
                    if isinstance(n, ast.ClassDef) and n.name == 'SHSBench')
    old = bench(T07 / 'lab_dvp8um/shs_physical2400.py')
    new = bench(T07 / 'hardware/bench.py')
    for name in ('__enter__', '__exit__', 'capture'):
        assert functions(old)[name] == functions(new)[name]
    init = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    required = {arg.arg for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults)
                if default is None}
    assert required == {'machine_config', 'phase_sdk', 'phase_lut'}
    assert functions(old)['__init__'] != functions(new)['__init__']
    # Prevent claiming the original 4-positional constructor can simply be swapped.
    calls = [n for n in ast.walk(tree(T12 / 'run_layerwise.py'))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'SHSBench']
    assert len(calls) == 1 and len(calls[0].args) == 4 and not calls[0].keywords
