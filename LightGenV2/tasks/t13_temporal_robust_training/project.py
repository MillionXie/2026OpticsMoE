"""Group-locked standalone project entry; also works outside the repository."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    condition = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python project.py plan|preflight|smoke|train|evaluate|theory|reference|infer [options]")
    action, arguments = sys.argv[1], sys.argv[2:]
    if "--group" in arguments or any(a.startswith("--group=") for a in arguments):
        raise SystemExit("This project is locked to one group; --group cannot be overridden")
    if action == "reference":
        script = ROOT / "teacher_reference" / "simulate.py"
        if not script.is_file():
            raise SystemExit("Build with --teacher-package to include pinned reference inference")
        return subprocess.call([sys.executable, "-I", str(script), *arguments], cwd=ROOT)
    if action == "infer":
        return subprocess.call([sys.executable, "-I", str(ROOT / "infer.py"), *arguments], cwd=ROOT)
    return subprocess.call([sys.executable, "-I", str(ROOT / "run.py"),
                            "--group", condition["group"], "--phase", action, *arguments], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
