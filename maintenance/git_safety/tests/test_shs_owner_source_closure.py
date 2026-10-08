"""Source/package checks only: never enter an SDK owner or open hardware."""
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
