"""Audit formal T06 Git dependencies against main; never replace runtime files.

The pinned source is independently bound to both existing, strict-loaded release
packages. A difference is NOT permission to overwrite: later lab-control fixes,
other consumers and historical profiles require separate review and CPU tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from plan_source_import import plan

SOURCE = "8e869473787f4ffceb2a6a77f4430b94c206f459"
TASK = "LightGenV2/tasks/t06_video_quality_assessment/"
BACKEND = "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/"
ENTRIES = [
    TASK + "multivideo.py", TASK + "lab_bundle.py", TASK + "lab_bench.py",
    BACKEND + "run.py",
    BACKEND + "tests/test_model_and_training.py",
]
PROFILES = [
    TASK + "configs/lightgen/temporal_multivideo16x4_formal.yaml",
    TASK + "configs/lightgen/temporal_multivideo9x4_formal.yaml",
    TASK + "configs/lightgen/spatial_single_video4_custom_conv_readout_1m.yaml",
]


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def audit(root: Path) -> dict:
    root = root.resolve()
    main = git(root, "rev-parse", "main").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    tree = set(git(root, "ls-tree", "-r", "--name-only", SOURCE).decode().splitlines())
    target_tree = set(git(root, "ls-tree", "-r", "--name-only", main).decode().splitlines())
    static = plan(root, SOURCE, ENTRIES)
    if static["missing_entries"]:
        raise RuntimeError("Missing pinned entry: " + str(static["missing_entries"]))
    paths = set(static["visited_paths"])
    queue, configs, contracts = list(PROFILES), set(), []
    while queue:
        path = queue.pop(0)
        if path in configs:
            continue
        if path not in tree or not path.startswith((TASK, BACKEND)):
            raise RuntimeError("Configuration escaped audited T06 closure: " + path)
        configs.add(path)
        raw = yaml.safe_load(git(root, "show", SOURCE + ":" + path))
        if not isinstance(raw, dict):
            raise RuntimeError("Expected a configuration mapping: " + path)
        parent = raw.get("base_config")
        if parent:
            resolved = (root / path).parent.joinpath(parent).resolve().relative_to(root).as_posix()
            queue.append(resolved)
        backend = raw.get("backend", {})
        if backend.get("config"):
            queue.append(backend["config"])
        for name, expected in backend.get("source_sha256", {}).items():
            dependency = BACKEND + name
            if dependency not in tree:
                raise RuntimeError("Missing declared backend: " + dependency)
            actual = hashlib.sha256(git(root, "show", SOURCE + ":" + dependency)).hexdigest()
            paths.add(dependency)
            contracts.append({"profile": path, "path": dependency,
                              "expected_sha256": expected, "source_sha256": actual,
                              "matches": expected == actual})
    paths.update(configs)
    rows = []
    for path in sorted(paths):
        data = git(root, "show", SOURCE + ":" + path)
        source_sha = hashlib.sha256(data).hexdigest()
        target = git(root, "show", main + ":" + path) if path in target_tree else None
        local = root / path
        rows.append({"path": path, "bytes": len(data), "source_sha256": source_sha,
                     "main_sha256": hashlib.sha256(target).hexdigest() if target is not None else None,
                     "main_relation": "missing" if target is None else
                         "identical" if target == data else "different_requires_review",
                     "working_sha256": hashlib.sha256(local.read_bytes()).hexdigest() if local.is_file() else None,
                     "python_compiles": bool(compile(data.decode("utf-8"), path, "exec")) if path.endswith(".py") else None})
    # Cross-task consumer inventory is evidence for compatibility review, not
    # proof that every possible dynamic import has been found.
    grep = subprocess.run(["git", "-C", str(root), "grep", "-n",
                           BACKEND.rstrip("/").replace("/", "."), main,
                           "--", "LightGenV2", ":!" + TASK.rstrip("/")],
                          capture_output=True, text=True)
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    if head != git(root, "rev-parse", "HEAD").decode().strip() or main != git(root, "rev-parse", "main").decode().strip():
        raise RuntimeError("Concurrent Git change; repeat audit")
    return {"schema_version": 1, "source_commit": SOURCE, "main_commit": main,
            "working_head": head, "entries": ENTRIES, "profiles": PROFILES,
            "files": rows, "profile_source_contracts": contracts,
            "cross_task_static_consumers": grep.stdout.splitlines(),
            "separate_later_source_review_required": [TASK + "adapt_measured_readout.py"],
            "static_imports_only": True, "dynamic_imports_complete": False,
            "runtime_tests_passed": False, "approved_for_overwrite": False,
            "working_files_changed": False, "data_or_weights_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier audit: " + str(args.output))
    report = audit(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(report["files"]), "counts": {
        relation: sum(row["main_relation"] == relation for row in report["files"])
        for relation in ("missing", "identical", "different_requires_review")},
        "profile_source_mismatches": sum(not row["matches"] for row in report["profile_source_contracts"]),
        "runtime_changed": False}))
