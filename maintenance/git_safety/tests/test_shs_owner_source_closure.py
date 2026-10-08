"""Source/package checks only: never enter an SDK owner or open hardware."""
import ast
import hashlib
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCOPE = 'LightGenV2/hardware_common/shs/'


def test_owner_and_display_keep_original_function_and_class_bodies():
    originals = {'phase_owner.py': '5c18f05c5fdeb5020cdfdd8583e4ef05dc4f1abc7b1bf9bae276bcfe001d902c',
                 'phase_display.py': '91163af118af02c1a8fb0f02c48dd541cd83dbff7285501e8c4d4c81de0eb502'}
    guard = ('if __package__:\n    from .phase_hdmi import PhaseHDMI, load_native\n'
             '    from .phase_display import DisplayOrigin\nelse:\n'
             '    from phase_hdmi import PhaseHDMI, load_native\n'
             '    from phase_display import DisplayOrigin\n')
    original_imports = 'from phase_hdmi import PhaseHDMI, load_native\nfrom phase_display import DisplayOrigin\n'
    for name, digest in originals.items():
        current = (ROOT/SCOPE/name).read_text(encoding='utf8')
        original = current.replace(guard, original_imports) if name == 'phase_owner.py' else current
        assert hashlib.sha256(original.encode()).hexdigest() == digest
        compile(current, name, 'exec')


def test_package_and_original_standalone_import_modes_open_no_hardware(monkeypatch):
    # Importing these modules must not call WinDLL or create a device owner.
    import ctypes
    def forbidden(*args, **kwargs):
        raise AssertionError('Hardware DLL must not load on module import')
    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    owner = importlib.import_module('LightGenV2.hardware_common.shs.phase_owner')
    display = importlib.import_module('LightGenV2.hardware_common.shs.phase_display')
    assert owner.PhaseOwner and display.DisplayOrigin
    assert display.DisplayOrigin().enabled is False
    monkeypatch.syspath_prepend(str(ROOT/SCOPE))
    standalone = importlib.import_module('phase_owner')
    assert standalone.PhaseOwner


def test_reference_requirements_keep_original_normalized_bytes():
    data = (ROOT/SCOPE/'requirements-gpu-tested-reference.txt').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(data).hexdigest() == '4b892cc1b076fb48a5f9d4dfc7700a8fa62e829ce23969d7a1b77b912d2fc1e3'


def test_historical_application_packager_uses_canonical_phase_source():
    source = (ROOT/'LightGenV2/tasks/t04_semantic_interaction/shs_package.py').read_text(encoding='utf8')
    assert "commit+':LightGenV2/hardware_common/shs/'+name" in source
    assert "commit+':ABO_Lab_SHS_8um/'+name" not in source
    original = source.replace("commit+':LightGenV2/hardware_common/shs/'+name", "commit+':ABO_Lab_SHS_8um/'+name")
    assert hashlib.sha256(original.encode()).hexdigest() == '8b937f52f1ed5e31d581584396680fe937d2001cf5261c736ff85667d222aba9'
    compile(source, 'shs_package.py', 'exec')


def test_shared_capture_uses_main_source_but_keeps_external_ownership_and_assets():
    source = (ROOT/'LightGenV2/tasks/t06_video_quality_assessment/lab_bench.py').read_text(encoding='utf8')
    tree = ast.parse(source)
    for name in ('capture', 'capture_staged'):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        text = ast.get_source_segment(source, fn)
        assert 'from LightGenV2.hardware_common.shs.slm_camera import Controller' in text
        assert 'sys.path.insert' not in text
        assert "lock=bench/'results/dual_jobs/ACTIVE.lock'" in text
        assert 'os.O_CREAT|os.O_EXCL|os.O_WRONLY' in text
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Controller']
        assert len(calls) == 1
        assert any(k.arg == 'config_base' and isinstance(k.value, ast.Name) and k.value.id == 'bench' for k in calls[0].keywords)
    original = source.replace('bench=Path(a.bench_root).resolve()\n from LightGenV2.hardware_common.shs.slm_camera import Controller',
                              'bench=Path(a.bench_root).resolve();sys.path.insert(0,str(bench))\n from slm_camera import Controller')
    original = original.replace('Controller(c,config_base=bench)', 'Controller(c)')
    original = original.replace('Controller(stage_config(c,a.stage,stages),config_base=bench)', 'Controller(stage_config(c,a.stage,stages))')
    assert hashlib.sha256(original.encode()).hexdigest() == '3c70a5fb8a1cb2be388084281999d57045f5c008f976cbccb3a3e6106f51217a'


def test_t06_bundle_contains_canonical_controller_dependency_closure():
    source = (ROOT/'LightGenV2/tasks/t06_video_quality_assessment/lab_bundle.py').read_text(encoding='utf8')
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    paths = []
    for n in fn.body:
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == 'paths' and isinstance(n.value, ast.ListComp):
            if isinstance(n.value.elt, ast.BinOp) and isinstance(n.value.elt.left, ast.Constant):
                # Parse literal source-path declarations without executing the builder.
                paths.extend(n.value.elt.left.value + item for item in ast.literal_eval(n.value.generators[0].iter))
    required = ['LightGenV2/hardware_common/shs/'+n for n in ('sdk.py','capture.py','slm_camera.py')]
    required += ['experiments/hardware_sdk/'+n for n in ('__init__.py','devices.py','drivers/__init__.py','drivers/meadowlark_pcie_slm.py','drivers/tucam_camera.py')]
    assert set(required) <= set(paths)
    assert all((ROOT/p).is_file() for p in required)
    additions = " paths += ['LightGenV2/hardware_common/shs/'+name for name in ('sdk.py','capture.py','slm_camera.py')]\n paths += ['experiments/hardware_sdk/'+name for name in ('__init__.py','devices.py','drivers/__init__.py','drivers/meadowlark_pcie_slm.py','drivers/tucam_camera.py')]\n"
    assert hashlib.sha256(source.replace(additions, '').encode()).hexdigest() == '833bb16a95403e5ee5114fda4da82659d9f652a63d4d42f3ffb1f6da6390d65c'
