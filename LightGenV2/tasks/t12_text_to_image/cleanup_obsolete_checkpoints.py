"""Explicit T12-only checkpoint cleanup; reports/data/worktrees are never deleted."""
import argparse
import hashlib
import json
from pathlib import Path


OBSOLETE = (
    "abo_audited_v2_large_pilot", "abo_audited_v2_large_adapt1000", "abo_audited_v2_large_joint3000",
    "abo_audited_v2_small_pilot", "abo_audited_v2_small_adapt1000", "abo_audited_v2_small_joint3000",
    "abo_audited_v2_small_joint_continue2000", "abo_audited_v2_small_detail_distill800",
    "abo_audited_v2_small_detail_gan1600", "abo_audited_v2_small_pruned996m_detail2000",
    "abo_audited_v2_small_pruned9p96m_gentle1000", "abo_audited_v2_small_14p72m_decoder_detail1800",
    "abo_unified_2layer_282m_v1", "abo_unified_expanded_143m_true256_v1",
    "abo_unified_expanded_143m_true256_v2", "abo_unified_expanded_qwenmini_9m_256_sourcegate_v1",
    "abo_unified_expanded_qwenmini_9m_256_v1", "abo_unified_expanded_qwenmini_9m_v1",
    "abo_unified_expanded_qwenmini_9m_v2", "abo_unified_expanded_qwenmini_9m_v3",
    "abo_unified_qwenmini2_optical_13m_v1", "abo_unified_qwenmini_143m_true256_alpha_v2",
    "abo_unified_qwenmini_143m_true256_v1", "abo_unified_qwenmini_143m_v1",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--sealed-reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.runs_root.resolve(strict=True)
    if root.name != "runs" or root.parent.name != "t12_assets":
        raise ValueError("Only the explicit legacy t12_assets/runs namespace is supported")
    # Cleanup is forbidden until both self-describing replacements passed audit.
    for name in ("small", "large"):
        reference = args.sealed_reference_dir/name
        if not reference.with_suffix(".pt").is_file():
            raise ValueError("Missing sealed reference")
        audit = json.loads((args.sealed_reference_dir/(name+"_audit.json")).read_text())
        if not audit.get("audit_passed"):
            raise ValueError("Sealed reference audit failed")
    active = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            active.append((proc/"cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (OSError, PermissionError):
            pass
    records, skipped = [], []
    for name in OBSOLETE:
        folder = root/name
        if not folder.exists():
            continue
        if folder.is_symlink() or folder.resolve() != folder:
            raise ValueError("Symlink or unexpected resolved run target")
        if any(str(folder) in command for command in active):
            skipped.append({"run": str(folder), "reason": "active process reference"})
            continue
        for path in sorted(folder.rglob("*.pt")):
            if path.is_symlink() or not path.resolve().is_relative_to(folder):
                raise ValueError("Unsafe checkpoint target")
            records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"root": str(root), "execute": args.execute, "files": records, "skipped": skipped,
              "bytes": sum(r["bytes"] for r in records), "deleted": 0,
              "retained": "all metrics/logs/images, current sealed references, datasets, baselines, other tasks"}
    args.output.write_text(json.dumps(result, indent=2))
    if args.execute:
        for record in records:
            path = Path(record["path"])
            if sha(path) != record["sha256"]:
                raise RuntimeError("Checkpoint changed after inventory")
            path.unlink()
            result["deleted"] += 1
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("files",)}, indent=2))


if __name__ == "__main__":
    main()
