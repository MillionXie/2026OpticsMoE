"""Build a self-contained ABO source/data/checkpoint handoff from committed code.

Keep the complete tracked compatibility source tree: selecting only directly
imported experiment directories is unsafe for transitive and dynamic imports.
Large runtime assets are explicitly allowlisted separately, never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

TASK = Path("LightGenV2/tasks/t08_abo_image_text_retrieval")
WARMSTART = Path("experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5/runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt")
RUNS = ("optical_router_moe_dc20_seed42", "optical_router_moe_dc20_kd1_balance1_seed42")
SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".sh", ".ps1"}
EXCLUDED = {"runs", "releases", "lab_bundles", "hardware_sessions", "artifacts", "vendor_sdk", "__pycache__", ".git", "reports", "data"}


def source_paths(root: Path) -> list[Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    return sorted(Path(n) for n in names if n and Path(n).suffix in SUFFIXES
                  and not EXCLUDED.intersection(Path(n).parts)
                  and (Path(n).parts[0] in {"LightGenV2", "experiments", "opticalmoe"}
                       or len(Path(n).parts) == 1))


def build(root: Path, assets: Path, output: Path) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    if dirty.strip():
        raise RuntimeError("Commit tracked source edits before building a release")
    files = {p: root / p for p in source_paths(root)}
    required = [Path("data/abo_easy100_dataset_20260906"), WARMSTART,
                TASK / "runs/shared/abo_easy100_qwen64"]
    required += [TASK / "runs/simulation" / run for run in RUNS]
    for relative in required:
        path = assets / relative
        if not path.exists():
            raise FileNotFoundError(path)
        for item in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if item.is_file() and "__pycache__" not in item.parts:
                files[item.relative_to(assets)] = item
    for run in RUNS:
        for name in ("best_checkpoint.pt", "final_report.json"):
            if TASK / "runs/simulation" / run / name not in files:
                raise FileNotFoundError(f"Incomplete run {run}: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "source_commit": commit,
                "task": "ABO easy100 image-to-title optical Router Top-2",
                "source_policy": "complete tracked compatibility sources/configs; allowlisted runtime assets",
                "external_dependency": "Qwen3-VL-Embedding-2B: supply --model-path to handoff prepare",
                "files": []}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for relative, path in sorted(files.items()):
            data = path.read_bytes()
            archive.writestr(relative.as_posix(), data)
            manifest["files"].append({"path": relative.as_posix(), "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    digest = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    return {"zip": str(output), "sha256": digest, "bytes": output.stat().st_size,
            "file_count": len(files), "source_commit": commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[3], args.asset_root.resolve(),
                           args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
