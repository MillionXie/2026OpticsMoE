"""Exercise the published HOLOEYE API-version contract using only a fake SDK."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch


SOURCE = "experiments/hardware_sdk/devices.py"


def check(root: Path, commit: str) -> dict:
    source = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{SOURCE}"])
    module = types.ModuleType("holoeye_git_contract")
    module.__file__ = str(root / SOURCE)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    with tempfile.TemporaryDirectory(prefix="holoeye_fake_sdk_") as temporary:
        base = Path(temporary)
        library = "holoeye_slmdisplaysdk.dll" if sys.platform.startswith("win") else "libholoeye_slmdisplaysdk.so"
        (base / library).touch()  # Never loaded: the native runtime is mocked.
        default = module.build_slm({"driver": "holoeye", "sdk_path": str(base)}, base)
        assert default.sdk_api_version == 5
        configured = module.build_slm({"driver": "holoeye", "sdk_path": str(base),
                                      "binary_folder": str(base), "sdk_api_version": 3,
                                      "expected_resolution_wh": [1920, 1080]}, base)
        calls = []
        class FakeSLM:
            width_px, height_px = 1920, 1080
            refreshrate_hz, pixelsize_um = 60, 8
            def __init__(self, *, binaryFolder):
                assert binaryFolder == str(base)
            def requiresVersion(self, version):
                calls.append(("requiresVersion", version)); return version == 3
            def open(self): calls.append(("open",)); return 0
            def close(self): calls.append(("close",))
        fake = types.SimpleNamespace(SLMInstance=FakeSLM,
                                     ErrorCode=types.SimpleNamespace(NoError=0))
        original_path = sys.path[:]
        try:
            with patch.object(importlib, "import_module", return_value=fake):
                configured.open()
                assert calls == [("requiresVersion", 3), ("open",)]
                assert configured.device_info()["sdk_api_version"] == 3
                configured.close()
                assert calls[-1] == ("close",)
                incompatible = module.HoloeyeSLM(base, binary_folder=base)
                try:
                    incompatible.open()
                except module.DeviceError as error:
                    assert "API version 5" in str(error)
                else:
                    raise AssertionError("API mismatch silently accepted")
                finally:
                    incompatible.close()
        finally:
            sys.path[:] = original_path
        assert sys.path == original_path
    return {"commit": commit, "source": SOURCE,
            "source_lf_sha256": hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
            "checks": ["API5 remains default", "explicit API3 reaches SDK", "API mismatch rejected",
                       "device metadata records configured version", "fake SDK closed"],
            "hardware_opened": False, "vendor_binary_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="main")
    args = parser.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[2], args.commit), indent=2))
