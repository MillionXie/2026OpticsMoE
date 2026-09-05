"""Rebuild the historical pre-reconciliation experiment audit.

The output describes project presence, storage, local Git state, and direct
Python dependencies.  It deliberately does not delete or move anything.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
OUT = (
    ROOT
    / "maintenance"
    / "storage"
    / "experiment_sync_audit_pre_reconciliation_20260905.csv"
)
SERVER_SIZES = ROOT / "maintenance" / "storage" / "server_experiment_sizes_20260904.csv"
SERVER_TRACKED = ROOT / "maintenance" / "storage" / "server_tracked_counts.csv"
SERVER_UNTRACKED = ROOT / "maintenance" / "storage" / "server_untracked_counts.csv"

IMPORT_RE = re.compile(r"(?:from|import)\s+experiments\.([A-Za-z0-9_]+)")
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "cache",
    "data",
    "deployment",
    "exports",
    "figures",
    "hardware_sessions",
    "lab_bundles",
    "logs",
    "results",
    "runs",
    "work",
}

INFRASTRUCTURE = {
    "hardware_sdk",
    "lab_lgvq",
    "lab_qwen",
    "qwen_optical_platform_handoff",
}
FORMAL_CURRENT = {
    "d2nn_mnist4_single_layer_17um_10cm_v2",
    "lgvq_four_stage_optical_electronic_109_no_attention_vqa",
    "qwen3_vl_2b_lgvq_single_metric_o2_16frame_54",
    "qwen3_vl_2b_lgvq_temporal_framecount_timing",
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff",
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval",
}
FORMAL_REFERENCE = {
    "qwen3_vl_2b_lgvq_linear_baseline",
    "qwen3_vl_2b_lgvq_o2_109_highalpha_vqa",
    "qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa",
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5",
}


def run_git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def read_two_column(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    rows: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("path,"):
            continue
        name, value = line.rsplit(",", 1)
        rows[name] = int(value)
    return rows


def project_for_path(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "experiments":
        return parts[1]
    return None


def local_status_counts() -> tuple[dict[str, int], dict[str, int]]:
    tracked: dict[str, int] = defaultdict(int)
    untracked: dict[str, int] = defaultdict(int)
    for line in run_git("status", "--porcelain=v1", "-uall"):
        project = project_for_path(line[3:].strip('"'))
        if not project:
            continue
        if line.startswith("??"):
            untracked[project] += 1
        else:
            tracked[project] += 1
    return dict(tracked), dict(untracked)


def local_sizes() -> tuple[dict[str, int], dict[str, int]]:
    sizes: dict[str, int] = {}
    counts: dict[str, int] = {}
    for project_dir in EXPERIMENTS.iterdir():
        if not project_dir.is_dir() or project_dir.name == "__pycache__":
            continue
        total = 0
        count = 0
        for base, dirs, files in os.walk(project_dir):
            dirs[:] = [name for name in dirs if name != ".git"]
            for name in files:
                path = Path(base) / name
                try:
                    total += path.stat().st_size
                    count += 1
                except OSError:
                    pass
        sizes[project_dir.name] = total
        counts[project_dir.name] = count
    return sizes, counts


def dependencies(projects: set[str]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for project in projects:
        project_dir = EXPERIMENTS / project
        if not project_dir.is_dir():
            continue
        for path in project_dir.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for dependency in IMPORT_RE.findall(text):
                if dependency in projects and dependency != project:
                    edges[project].add(dependency)
    return edges


def dependency_closure(edges: dict[str, set[str]], roots: set[str]) -> set[str]:
    seen: set[str] = set()
    queue = deque(roots)
    while queue:
        project = queue.popleft()
        for dependency in edges.get(project, set()):
            if dependency not in seen and dependency not in roots:
                seen.add(dependency)
                queue.append(dependency)
    return seen


def main() -> None:
    local_bytes, local_files = local_sizes()
    server_bytes = read_two_column(SERVER_SIZES)
    server_tracked = read_two_column(SERVER_TRACKED)
    server_untracked = read_two_column(SERVER_UNTRACKED)
    github = set(run_git("ls-tree", "-d", "--name-only", "origin/main:experiments"))
    local_tracked, local_untracked = local_status_counts()
    projects = (set(local_bytes) | set(server_bytes) | github) - {"__pycache__"}
    edges = dependencies(projects)
    reverse: dict[str, set[str]] = defaultdict(set)
    for project, deps in edges.items():
        for dependency in deps:
            reverse[dependency].add(project)
    closure = dependency_closure(edges, FORMAL_CURRENT | FORMAL_REFERENCE | INFRASTRUCTURE)

    rows = []
    for project in sorted(projects):
        if project in INFRASTRUCTURE:
            role = "infrastructure_or_handoff"
            action = "publish_and_keep"
        elif project in FORMAL_CURRENT:
            role = "formal_current"
            action = "publish_and_keep"
        elif project in FORMAL_REFERENCE:
            role = "formal_reference"
            action = "keep_reference"
        elif project in closure:
            role = "required_dependency"
            action = "keep_until_refactored"
        else:
            role = "owner_review"
            action = "audit_before_archive_or_delete"
        rows.append(
            {
                "project": project,
                "role_hint": role,
                "recommended_source_action": action,
                "local_present": project in local_bytes,
                "server_present": project in server_bytes,
                "github_tracked": project in github,
                "local_gib": f"{local_bytes.get(project, 0) / 2**30:.3f}",
                "server_gib": f"{server_bytes.get(project, 0) / 2**30:.3f}",
                "local_files": local_files.get(project, 0),
                "local_tracked_changes": local_tracked.get(project, 0),
                "local_untracked_files": local_untracked.get(project, 0),
                "server_tracked_changes": server_tracked.get(project, 0),
                "server_untracked_files": server_untracked.get(project, 0),
                "imports_projects": ";".join(sorted(edges.get(project, set()))),
                "imported_by_projects": ";".join(sorted(reverse.get(project, set()))),
                "safe_to_delete_whole_project_now": "no",
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} projects to {OUT}")


if __name__ == "__main__":
    main()
