"""No-device tests of the exact selection and pre-persistence guard functions."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

TASK = Path(__file__).resolve().parents[1]
PROFILE = TASK / "configs/lab/rank64_20261002.json"
spec = importlib.util.spec_from_file_location("lab_identity", TASK / "lab_checkpoint_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
source = ast.parse((TASK / "lab_shs_capture.py").read_text(encoding="utf-8"))
selected = [n for n in source.body if
            (isinstance(n, ast.FunctionDef) and n.name in ("checkpoint_identity", "signal_guard")) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GROUPS" for t in n.targets))]
namespace = {"Path": Path, "load_identity": identity.load_identity, "Image": Image}
exec(compile(ast.Module(body=selected, type_ignores=[]), "exact_capture_functions", "exec"), namespace)


def test_historical_default_is_still_rank16():
    for group, (filename, digest) in namespace["GROUPS"].items():
        row = namespace["checkpoint_identity"](group)
        assert row == {"group": group, "filename": filename, "sha256": digest,
                       "shared_readout_variant": "lowrank16"}


def test_rank64_requires_explicit_profile():
    for group in ("g2", "g5"):
        assert namespace["checkpoint_identity"](group, PROFILE) == identity.load_identity(PROFILE, group)
    with pytest.raises(ValueError):
        namespace["checkpoint_identity"]("g3", PROFILE)


@pytest.mark.parametrize("p99", [11, 14])
def test_dark_frame_never_persisted(tmp_path, p99):
    class Bench:
        rows = []
        out = tmp_path
        def capture(self, *args, save=True):
            assert save is False
            self.rows.append({"sample_id": "sample", "p99": p99})
            return np.zeros((1, 2, 2), dtype=np.uint8), {"phase": "synthetic"}
    with pytest.raises(RuntimeError, match="not persisted"):
        namespace["signal_guard"](Bench)().capture("stage", None, [None], ["sample"], "flip_v")
    assert not (tmp_path / "ccd").exists()


def test_valid_frame_saved_only_after_guard(tmp_path, monkeypatch):
    records = []
    monkeypatch.setitem(namespace, "write", lambda path, row: records.append((path, row)))
    class Bench:
        out = tmp_path
        def __init__(self):
            self.rows = []
        def capture(self, *args, save=True):
            assert save is False
            self.rows.append({"sample_id": "sample", "p99": 90})
            return np.full((1, 2, 2), 90, dtype=np.uint8), {"phase": "synthetic"}
    values, receipt = namespace["signal_guard"](Bench)().capture("stage", None, [None], ["sample"], "flip_v")
    assert values.shape == (1, 2, 2) and receipt["phase"] == "synthetic"
    assert (tmp_path / "ccd/stage/sample.png").exists()
    assert records[0][0] == tmp_path / "ccd/stage/sample.json"


def test_rank64_validation_precedes_device_import():
    main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    validation = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "checkpoint_identity")
    sdk_import = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.ImportFrom)
                      and n.module == "LightGenV2.tasks.t07_abo_image_retrieval.hardware.bench")
    assert validation < sdk_import


def test_capture_uses_only_canonical_control_source():
    text = ast.unparse(source)
    assert 'sys.path.insert' not in text
    assert 'four_image_flow' not in text and 'shs_physical2400' not in text
    assert 'machine_config=args.machine_config' in text
    assert 'phase_sdk=args.phase_sdk' in text and 'phase_lut=args.phase_lut' in text
    assert 'source_root /' in text and 'project / \'source/' not in text
