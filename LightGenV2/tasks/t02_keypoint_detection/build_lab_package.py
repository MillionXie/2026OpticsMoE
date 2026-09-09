"""Committed-source LSP handoff with explicit data and checkpoint dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

TASK = Path("LightGenV2/tasks/t02_keypoint_detection")
LEGACY = Path("experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16")
ASSETS = (
    Path("data/lsp_pose/lsp_dataset"), Path("data/lsp_pose/lspet_dataset"),
    TASK / "runs/simulation/moe_router_scale_dc20_no_shift_warmstart0713_seed42",
    TASK / "runs/simulation/moe_router_scale_dc20_seed42",
    TASK / "runs/simulation/moe_router_scale_dc20_no_shift_seed42",
    TASK / "runs/simulation/d2nn_matched_dc20_seed42",
    Path("experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/runs/shared_untrained_initialization.pt"),
    LEGACY / "runs/lsp_pose_vision2_hybrid/checkpoints/student_best_train_loss.pt",
    LEGACY / "runs/lsp_pose_optical_moe16_opt2/checkpoints/teacher_best_train_loss.pt",
)


def source_paths(root: Path) -> list[Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    excluded = {"runs", "releases", "lab_bundles", "hardware_sessions", "artifacts", "vendor_sdk", "data", "__pycache__"}
    suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".toml", ".sh", ".ps1"}
    return sorted(Path(n) for n in names if n and Path(n).suffix in suffixes
                  and not excluded.intersection(Path(n).parts)
                  and (Path(n).parts[0] in {"LightGenV2", "experiments", "opticalmoe"}
                       or len(Path(n).parts) == 1))


def build(root: Path, assets: Path, output: Path) -> dict:
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root).strip():
        raise RuntimeError("Commit tracked edits before packaging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    files = {p: root / p for p in source_paths(root)}
    files[Path("START_HERE.md")] = root / TASK / "HANDOFF.md"
    for relative in ASSETS:
        path = assets / relative
        if not path.exists():
            raise FileNotFoundError(path)
        for item in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if item.is_file() and "__pycache__" not in item.parts:
                files[item.relative_to(assets)] = item
    manifest = {"source_commit": commit, "task": "T02 LSP vision-only optical Router Top-2",
                "external_model": "Qwen3-VL-Embedding-2B reused from recipient; not duplicated",
                "source_policy": "complete tracked compatibility code/configs; explicit runtime allowlist",
                "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for relative, path in sorted(files.items()):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            archive.write(path, relative.as_posix(), compress_type=(zipfile.ZIP_STORED if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else zipfile.ZIP_DEFLATED))
            manifest["files"].append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest})
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n")
    return {"zip": str(output), "sha256": digest, "files": len(files), "bytes": output.stat().st_size, "commit": commit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[3], args.asset_root.resolve(), args.output.resolve()), indent=2))
