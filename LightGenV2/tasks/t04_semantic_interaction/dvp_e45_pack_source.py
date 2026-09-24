"""Package the exact e45 training branch's import closure for DVP deployment."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import zipfile

from .build_lab_package import source_closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[3]
    files = {path.relative_to(root).as_posix() for path in source_closure()}
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", args.commit, "LightGenV2", "experiments"],
        cwd=root, text=True,
    ).splitlines()
    files.update(
        rel for rel in tracked if rel.endswith(".yaml")
        and not any(part in {"runs", "data", "dataset", "vendor_sdk"} for part in Path(rel).parts)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted(files):
            content = subprocess.check_output(["git", "show", f"{args.commit}:{rel}"], cwd=root)
            archive.writestr(rel, content)
    print(f"PACKAGED {len(files)} files -> {output}")


if __name__ == "__main__":
    main()
