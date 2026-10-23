"""Read-only source/architecture inventory. This module cannot launch training.

Run on either host with CUDA_VISIBLE_DEVICES empty. Optional --output saves
new review evidence in this task only; it refuses to overwrite a prior file.
--server-inventory queries nvidia-smi without initializing a CUDA context.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from .models import TASK_IDS, build_model, parameter_counts

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[2]


def command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8",
                                errors="replace", capture_output=True, timeout=25)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(server_inventory: bool = False) -> dict:
    import torch
    config_path = TASK_DIR / "configs/suite_proposal.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if [t["id"] for t in config["tasks"]] != list(TASK_IDS):
        raise ValueError("Exactly the nine reviewed table rows, in table order, are required")
    if config["architecture_approved"] or config["training_authorized"]:
        raise ValueError("This is a preparation-only profile; do not rewrite its approval flags")
    authorization_path = TASK_DIR / "configs/execution_authorization_20261023.json"
    authorization = (json.loads(authorization_path.read_text(encoding="utf-8"))
                     if authorization_path.is_file() else None)
    pending_gates = list(config["launch_gates"])
    if authorization and authorization.get("training_authorized_after_required_preflight"):
        pending_gates = [gate for gate in pending_gates if gate not in {
            "user_accepts_architecture_and_scope",
            "explicit_user_authorization_before_any_GPU_training_or_GPU_smoke"}]
    source_paths = {config["reference_table"]}
    for task in config["tasks"]:
        prefix = Path("LightGenV2/tasks") / task["task_dir"]
        source_paths.add((prefix / "README.md").as_posix())
        source_paths.update((prefix / s).as_posix() for s in task["baseline_sources"])
    sources, missing_evidence = [], []
    for relative in sorted(source_paths):
        path = ROOT / relative
        if not path.is_file():
            # Historical measured tables/reports are retained external artifacts,
            # not source dependencies installed by Git on every host.
            if '/reports/' in relative:
                missing_evidence.append(relative)
                continue
            raise FileNotFoundError(f"Baseline reference missing: {relative}")
        sources.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        models = [{"row": task["row"], "id": task["id"],
                   "parameters": parameter_counts(build_model(task["id"]))}
                  for task in config["tasks"]]
    result = {
        "status": "local_architecture_preparation_only_not_ready_to_train",
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_branch": command(["git", "branch", "--show-current"]),
        "git_tracked_status": command(["git", "status", "--porcelain", "--untracked-files=no"]),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "cuda_initialized": torch.cuda.is_initialized(),
        "proposal_sha256": sha256(config_path),
        "model_source_sha256": sha256(TASK_DIR / "models.py"),
        "models": models, "baseline_source_files": sources,
        "historical_evidence_not_installed": missing_evidence,
        "remote_source_verified": False, "remote_data_verified": False,
        "local_Git_GitHub_server_synchronized": False,
        "real_dataset_adapters_implemented": False,
        "partial_dataset_adapters": {
            "abo_easy100": "implemented_CPU_contract_tested_original_data_not_available_locally",
            "other_tasks": "pending",
        },
        "formal_training_started": False, "GPU_smoke_started": False,
        "GPU_scheduler_implemented": False,
        "execution_authorization": authorization,
        "effective_execution_profile": json.loads((TASK_DIR / 'configs/execution_single_seed30_20261023.json').read_text(encoding='utf-8')),
        "authorization_sha256": sha256(authorization_path) if authorization else None,
        "pending_gates": pending_gates,
    }
    if server_inventory:
        result["read_only_gpu_inventory"] = command([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader"])
        result["read_only_compute_processes"] = command([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader"])
        result["inventory_does_not_reserve_a_gpu"] = True
    return result


def save_new_report(path: Path, report: dict) -> None:
    path = path.resolve()
    owners = [TASK_DIR / "reports", TASK_DIR / "runs/smoke"]
    if not any(path.is_relative_to(p.resolve()) for p in owners):
        raise ValueError("Review evidence must stay in this task's reports or runs/smoke")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-inventory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.server_inventory)
    if args.output:
        save_new_report(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "baseline_source_files"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
