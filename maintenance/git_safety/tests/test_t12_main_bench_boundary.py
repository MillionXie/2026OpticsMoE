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


def test_new_adapter_changes_only_two_reviewed_binding_fragments():
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.main_layerwise import (
        adapted_source, OLD_ARGUMENT, OLD_BINDING, NEW_BINDING)
    original = (T12 / 'run_layerwise.py').read_text(encoding='utf8').replace('\r\n', '\n')
    changed = adapted_source()
    assert changed == original.replace(OLD_ARGUMENT, '').replace(OLD_BINDING, NEW_BINDING)
    compile(changed, 'fixture', 'exec')
    assert 'a.abo_project' not in changed
    # The model execution and scientific stage body must not be modernized here.
    assert ".cuda().eval().requires_grad_(False)" in changed
    assert "SHSBench(a.output,400,240,{})" in changed


def test_inspection_creates_no_output_or_device_import(tmp_path, monkeypatch):
    import argparse
    import json
    import sys
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um import main_layerwise as adapter
    project = tmp_path / 'project'; (project / 'assets').mkdir(parents=True)
    (project / 'assets/contract.json').write_text(json.dumps(
        {'checkpoint_sha256': adapter.FORMAL_SHA, 'counted_parameters': 17026642}))
    (project / 'assets/small.pt').write_bytes(b'fixture only; hash is mocked')
    for name in ('machine.json', 'phase.lut'):
        (tmp_path / name).write_bytes(b'fixture')
    sdk = tmp_path / 'phase_sdk'; sdk.mkdir()
    (sdk/'Blink_C_wrapper.dll').write_bytes(b'fixture')
    reuse = tmp_path / 'reuse'; reuse.mkdir()
    out = tmp_path / 'output'
    args = argparse.Namespace(project=project, output=out, reuse=reuse,
                             machine_config=tmp_path/'machine.json',
                             phase_sdk=sdk, phase_lut=tmp_path/'phase.lut',
                             amplitude_sdk=None)
    monkeypatch.setattr(adapter, 'digest', lambda p: adapter.FORMAL_SHA)
    before = set(sys.modules)
    result = adapter.inspect(args)
    assert not out.exists() and not result['devices_opened'] and not result['model_loaded']
    assert 'torch' not in set(sys.modules) - before
    assert not any('hardware.bench' in p for p in set(sys.modules) - before)
    out.mkdir()
    import pytest
    with pytest.raises(FileExistsError):
        adapter.inspect(args)


def test_inspection_rejects_other_model_contract(tmp_path):
    import argparse
    import json
    import pytest
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.main_layerwise import inspect
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/contract.json').write_text(json.dumps(
        {'checkpoint_sha256': '0'*64, 'counted_parameters': 9958098}))
    with pytest.raises(ValueError, match='formal 17M'):
        inspect(argparse.Namespace(project=tmp_path))


def test_main_train_binding_preserves_original_selection_guard():
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.main_layerwise import adapted_source
    source = adapted_source(True)
    compile(source, 'train_fixture', 'exec')
    assert "assert len(dataset)==20736" in source
    assert "selection['test_product_overlap']==0" in source
    assert "selection['test_source_hash_overlap']==0" in source
    assert "assert a.max_samples is None" in source
    assert "'train_index'" in source
    assert 'a.abo_project' not in source


def test_train_inspect_rejects_test_overlap_and_invalid_indices(tmp_path):
    import argparse
    import json
    import pytest
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.main_layerwise import inspect
    selection = tmp_path/'selection.json'
    args = argparse.Namespace(split='train', selection=selection, max_samples=None)
    good = {'split':'train','test_product_overlap':0,'test_source_hash_overlap':0,'indices':[0,1]}
    for update, message in [({'test_product_overlap':1}, 'TEST overlap'),
                            ({'test_source_hash_overlap':1}, 'TEST overlap'),
                            ({'split':'test'}, 'TEST overlap'),
                            ({'indices':[0,0]}, 'indices'),
                            ({'indices':[-1]}, 'indices'),
                            ({'indices':[20736]}, 'indices'),
                            ({'indices':[True]}, 'indices'),
                            ({'indices':[]}, 'indices')]:
        selection.write_text(json.dumps({**good, **update}))
        with pytest.raises(ValueError, match=message):
            inspect(args)
    args.max_samples = 1
    with pytest.raises(ValueError, match='no --max-samples'):
        inspect(args)


def test_write_helper_import_does_not_launch_legacy_capture():
    # Formal runner only imports this function; run_full.main is not invoked.
    imported = tree(T12/'run_full.py')
    assert not any(isinstance(n, ast.Import) and any(a.name in ('four_image_flow','shs_physical2400')
                                                  for a in n.names) for n in imported.body)
    for n in imported.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            raise AssertionError('run_full gained an import-time side effect')
