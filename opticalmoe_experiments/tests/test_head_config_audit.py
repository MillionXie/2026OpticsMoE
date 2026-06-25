import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_head_config_audit_script_passes():
    script = ROOT / "scripts" / "audit_head_config_fields.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(ROOT)],
        cwd=str(ROOT.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout.lower()
