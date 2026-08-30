from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Support module and direct script execution.
    from .settings import METHODS, TASKS, Settings, implementation_sha256, load_settings
except ImportError:  # pragma: no cover
    from settings import (  # type: ignore[no-redef]
        METHODS,
        TASKS,
        Settings,
        implementation_sha256,
        load_settings,
    )


MODULE = (
    "experiments."
    "qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.run"
)
RESULT_FORMAT = "p12-p11-downstream-result-v1"
STATE_FORMAT = "p12-multigpu-queue-v1"
_FILE_HASH_CACHE: dict[Path, tuple[int, int, str]] = {}


def _implementation_sha256() -> str:
    return implementation_sha256()


@dataclass(frozen=True, order=True)
class JobKey:
    task: str
    method: str
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.task}__{self.method}__seed_{self.seed}"


@dataclass(frozen=True)
class GPUDevice:
    index: int
    uuid: str


@dataclass
class ActiveJob:
    key: JobKey
    gpu_index: int
    gpu_uuid: str
    pid: int
    attempt: int
    started_at_unix: float
    process: subprocess.Popen[bytes] | None = None
    stdout_handle: Any = None
    stderr_handle: Any = None


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from error
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("list must contain non-negative integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("list must not contain duplicates")
    return parsed


def build_job_matrix(
    seeds: Iterable[int], adaptation_seeds: Iterable[int] | None = None
) -> list[JobKey]:
    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds or any(seed < 0 for seed in unique_seeds):
        raise ValueError("at least one non-negative seed is required")
    selected_adaptation_seeds = (
        unique_seeds
        if adaptation_seeds is None
        else tuple(dict.fromkeys(int(seed) for seed in adaptation_seeds))
    )
    unknown = set(selected_adaptation_seeds) - set(unique_seeds)
    if not selected_adaptation_seeds or unknown:
        raise ValueError(
            "adaptation_seeds must be a non-empty subset of seeds; "
            f"unknown={sorted(unknown)}"
        )
    # Common starts are intentionally first. Updating jobs may nevertheless
    # launch as soon as their own task/seed dependency completes.
    common = [JobKey(task, "noft", seed) for seed in unique_seeds for task in TASKS]
    updates = [
        JobKey(task, method, seed)
        for seed in selected_adaptation_seeds
        for task in TASKS
        for method in METHODS
        if method != "noft"
    ]
    return common + updates


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    cached = _FILE_HASH_CACHE.get(resolved)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[resolved] = (*signature, value)
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _update_json_atomic(path: Path, values: Mapping[str, Any]) -> None:
    current = _read_json(path) or {}
    current.update(values)
    _write_json_atomic(path, current)


def resolved_settings(config: Path, key: JobKey) -> Settings:
    return load_settings(config, task=key.task, method=key.method, seed=key.seed)


def completion_reason(settings: Settings) -> tuple[bool, str]:
    """Strictly validate the terminal artifact; checkpoints are never terminal."""
    path = settings.output_dir / "result.json"
    result = _read_json(path)
    if result is None:
        return False, "result.json missing or invalid"
    implementation_digest = _implementation_sha256()
    expected: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": "complete",
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "head_only_epochs": 50,
        "epochs_completed_this_run": 50,
        "inherited_pipeline_epochs": 50 if settings.method == "noft" else 100,
        "adaptation_epochs": 0 if settings.method == "noft" else 50,
        "config_digest": _json_digest(
            {
                "settings": settings.to_dict(),
                "implementation_sha256": implementation_digest,
            }
        ),
        "implementation_sha256": implementation_digest,
        "source_checkpoint_sha256": settings.paths.source_backbone_sha256,
    }
    mismatches = {
        name: (result.get(name), required)
        for name, required in expected.items()
        if result.get(name) != required
    }
    if mismatches:
        return False, f"result identity/protocol mismatch: {mismatches}"
    best_epoch = result.get("best_epoch")
    minimum_best_epoch = 1 if settings.method == "noft" else 0
    if not isinstance(best_epoch, int) or not minimum_best_epoch <= best_epoch <= 50:
        return False, f"best_epoch is missing or outside {minimum_best_epoch}..50"
    primary_name = settings.task_settings.primary_metric
    if result.get("primary_metric") != primary_name:
        return False, "primary_metric identity mismatch"
    primary = result.get("test", {}).get("normal", {}).get(primary_name)
    if not isinstance(primary, (int, float)) or not math.isfinite(float(primary)):
        return False, "finite test.normal primary metric missing"
    for hash_name in (
        "source_checkpoint_sha256",
        "dataset_manifest_sha256",
        "implementation_sha256",
    ):
        value = result.get(hash_name)
        if not isinstance(value, str) or len(value) != 64:
            return False, f"{hash_name} missing or malformed"
    if (
        not settings.paths.source_backbone.is_file()
        or _sha256_file(settings.paths.source_backbone)
        != settings.paths.source_backbone_sha256
    ):
        return False, "current P11 source checkpoint checksum mismatch"
    manifest = _read_json(
        settings.output_dir / "manifests" / f"{settings.task}_splits.json"
    )
    if manifest is None or manifest.get("manifest_sha256") != result.get(
        "dataset_manifest_sha256"
    ):
        return False, "saved dataset manifest checksum identity mismatch"
    common = settings.paths.common_start_checkpoint
    if not common.is_file() or common.stat().st_size == 0:
        return False, "common_start.pt missing or empty"
    expected_hash = result.get("common_start_sha256")
    if not isinstance(expected_hash, str) or _sha256_file(common) != expected_hash:
        return False, "common_start.pt checksum mismatch"
    return True, "complete"


def dependency_ready(config: Path, key: JobKey) -> tuple[bool, str]:
    if key.method == "noft":
        return True, "no dependency"
    common = resolved_settings(config, JobKey(key.task, "noft", key.seed))
    complete, reason = completion_reason(common)
    if not complete:
        return False, f"waiting for NoFT/common_start: {reason}"
    return True, "common start complete"


def _csv_rows(output: str) -> list[list[str]]:
    return [
        [column.strip() for column in row]
        for row in csv.reader(output.splitlines())
        if row and any(column.strip() for column in row)
    ]


def query_gpu_inventory(
    *, nvidia_smi: str = "nvidia-smi", runner=subprocess.run
) -> tuple[dict[int, GPUDevice], dict[str, list[dict[str, Any]]]]:
    gpu_result = runner(
        [nvidia_smi, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices: dict[int, GPUDevice] = {}
    for row in _csv_rows(gpu_result.stdout):
        if len(row) < 2:
            raise RuntimeError(f"Malformed nvidia-smi GPU row: {row}")
        device = GPUDevice(index=int(row[0]), uuid=row[1])
        devices[device.index] = device

    app_result = runner(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    applications: dict[str, list[dict[str, Any]]] = {}
    for row in _csv_rows(app_result.stdout):
        if len(row) < 2 or row[0] in {"N/A", "[N/A]", ""}:
            continue
        try:
            pid = int(row[1])
        except ValueError:
            continue
        applications.setdefault(row[0], []).append(
            {
                "pid": pid,
                "process_name": row[2] if len(row) > 2 else "unknown",
                "used_gpu_memory_mib": (
                    int(row[3]) if len(row) > 3 and row[3].isdigit() else None
                ),
            }
        )
    return devices, applications


def free_requested_gpus(
    requested: Sequence[int],
    devices: Mapping[int, GPUDevice],
    applications: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reserved_indices: Iterable[int] = (),
) -> list[GPUDevice]:
    missing = [index for index in requested if index not in devices]
    if missing:
        raise RuntimeError(f"Requested GPU indices do not exist: {missing}")
    reserved = set(reserved_indices)
    return [
        devices[index]
        for index in requested
        if index not in reserved and not applications.get(devices[index].uuid)
    ]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class QueueLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> "QueueLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            previous = _read_json(self.path) or {}
            previous_pid = int(previous.get("pid", -1))
            if _pid_alive(previous_pid):
                raise RuntimeError(
                    f"Another P12 queue manager is active (pid={previous_pid})"
                )
            self.path.unlink(missing_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        payload = json.dumps(
            {"pid": os.getpid(), "hostname": socket.gethostname(), "started": time.time()}
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


class MultiGPUQueue:
    def __init__(
        self,
        *,
        config: Path,
        gpu_indices: Sequence[int],
        seeds: Sequence[int],
        python: Path,
        repo_root: Path,
        adaptation_seeds: Sequence[int] | None = None,
        poll_seconds: float = 20.0,
        max_retries: int = 2,
        nvidia_smi: str = "nvidia-smi",
    ) -> None:
        self.config = config.resolve()
        self.gpu_indices = tuple(gpu_indices)
        self.seeds = tuple(seeds)
        self.adaptation_seeds = (
            self.seeds if adaptation_seeds is None else tuple(adaptation_seeds)
        )
        self.python = python.resolve()
        self.repo_root = repo_root.resolve()
        self.poll_seconds = float(poll_seconds)
        self.max_retries = int(max_retries)
        self.nvidia_smi = nvidia_smi
        if not self.python.is_file():
            raise FileNotFoundError(f"Python executable not found: {self.python}")
        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"Repository root not found: {self.repo_root}")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.jobs = build_job_matrix(self.seeds, self.adaptation_seeds)
        first = resolved_settings(self.config, self.jobs[0])
        self.queue_dir = first.paths.output_root / "queue"
        self.state_path = self.queue_dir / "queue_state.json"
        self.lock_path = self.queue_dir / "queue.lock"
        self.attempts: dict[str, int] = {job.slug: 0 for job in self.jobs}
        self.active: dict[str, ActiveJob] = {}
        self.terminal_failures: dict[str, str] = {}
        self._restore_state()

    def _restore_state(self) -> None:
        state = _read_json(self.state_path)
        if state is None or state.get("format") != STATE_FORMAT:
            return
        if (
            state.get("config") != str(self.config)
            or tuple(state.get("seeds", ())) != self.seeds
            or tuple(state.get("adaptation_seeds", ())) != self.adaptation_seeds
        ):
            return
        old_jobs = state.get("jobs", {})
        for job in self.jobs:
            row = old_jobs.get(job.slug, {}) if isinstance(old_jobs, dict) else {}
            self.attempts[job.slug] = int(row.get("attempts", 0))
            pid = int(row.get("pid") or -1)
            if row.get("status") == "running" and _pid_alive(pid):
                self.active[job.slug] = ActiveJob(
                    key=job,
                    gpu_index=int(row["gpu_index"]),
                    gpu_uuid=str(row.get("gpu_uuid", "unknown")),
                    pid=pid,
                    attempt=self.attempts[job.slug],
                    started_at_unix=float(row.get("started_at_unix", time.time())),
                )
            elif row.get("status") == "failed":
                self.terminal_failures[job.slug] = str(
                    row.get("reason", "previous queue exhausted retries")
                )

    def _state(self, *, note: str = "running") -> dict[str, Any]:
        rows: dict[str, Any] = {}
        counts = {name: 0 for name in ("complete", "running", "pending", "blocked", "failed")}
        for job in self.jobs:
            settings = resolved_settings(self.config, job)
            complete, reason = completion_reason(settings)
            active = self.active.get(job.slug)
            if complete:
                status = "complete"
            elif active is not None:
                status = "running"
                reason = "process active"
            elif job.slug in self.terminal_failures:
                status = "failed"
                reason = self.terminal_failures[job.slug]
            else:
                ready, dependency = dependency_ready(self.config, job)
                status = "pending" if ready else "blocked"
                reason = dependency
            counts[status] += 1
            row: dict[str, Any] = {
                **asdict(job),
                "status": status,
                "reason": reason,
                "attempts": self.attempts[job.slug],
                "run_dir": str(settings.output_dir),
            }
            if active is not None:
                row.update(
                    {
                        "pid": active.pid,
                        "gpu_index": active.gpu_index,
                        "gpu_uuid": active.gpu_uuid,
                        "started_at_unix": active.started_at_unix,
                    }
                )
            rows[job.slug] = row
        return {
            "format": STATE_FORMAT,
            "updated_at_unix": time.time(),
            "hostname": socket.gethostname(),
            "manager_pid": os.getpid(),
            "note": note,
            "config": str(self.config),
            "repo_root": str(self.repo_root),
            "python": str(self.python),
            "requested_gpus": list(self.gpu_indices),
            "seeds": list(self.seeds),
            "adaptation_seeds": list(self.adaptation_seeds),
            "max_retries": self.max_retries,
            "counts": counts,
            "jobs": rows,
        }

    def _persist(self, *, note: str = "running") -> dict[str, Any]:
        state = self._state(note=note)
        _write_json_atomic(self.state_path, state)
        return state

    def _close_handles(self, active: ActiveJob) -> None:
        for handle in (active.stdout_handle, active.stderr_handle):
            if handle is not None:
                handle.close()

    def _poll_active(self) -> None:
        for slug, active in list(self.active.items()):
            if active.process is not None:
                return_code = active.process.poll()
                alive = return_code is None
            else:
                alive = _pid_alive(active.pid)
                return_code = None
            if alive:
                continue
            self._close_handles(active)
            self.active.pop(slug)
            settings = resolved_settings(self.config, active.key)
            complete, reason = completion_reason(settings)
            if complete:
                _update_json_atomic(
                    settings.output_dir / "process.json",
                    {
                        "status": "complete",
                        "pid": active.pid,
                        "gpu_index": active.gpu_index,
                        "attempt": active.attempt,
                        "finished_at_unix": time.time(),
                    },
                )
                print(f"[P12 queue] complete {slug}", flush=True)
                continue
            if self.attempts[slug] > self.max_retries:
                self.terminal_failures[slug] = (
                    f"exhausted {self.attempts[slug]} attempt(s); exit={return_code}; {reason}"
                )
                _update_json_atomic(
                    settings.output_dir / "process.json",
                    {
                        "status": "failed",
                        "pid": active.pid,
                        "gpu_index": active.gpu_index,
                        "attempt": active.attempt,
                        "exit_code": return_code,
                        "reason": reason,
                        "finished_at_unix": time.time(),
                    },
                )
                print(f"[P12 queue] terminal failure {slug}: {reason}", flush=True)
            else:
                _update_json_atomic(
                    settings.output_dir / "process.json",
                    {
                        "status": "retry_pending",
                        "pid": active.pid,
                        "gpu_index": active.gpu_index,
                        "attempt": active.attempt,
                        "exit_code": return_code,
                        "reason": reason,
                        "finished_at_unix": time.time(),
                    },
                )
                print(
                    f"[P12 queue] retry pending {slug}: exit={return_code}; {reason}",
                    flush=True,
                )

    def _next_ready_job(self) -> JobKey | None:
        for job in self.jobs:
            if job.slug in self.active or job.slug in self.terminal_failures:
                continue
            complete, _ = completion_reason(resolved_settings(self.config, job))
            if complete:
                continue
            ready, _ = dependency_ready(self.config, job)
            if ready:
                return job
        return None

    def _launch(self, job: JobKey, gpu: GPUDevice) -> bool:
        # Re-query immediately before Popen to close the common stale-snapshot
        # failure mode. The queue never launches onto a compute-occupied UUID.
        devices, applications = query_gpu_inventory(nvidia_smi=self.nvidia_smi)
        free = free_requested_gpus(
            [gpu.index],
            devices,
            applications,
            reserved_indices=(active.gpu_index for active in self.active.values()),
        )
        if not free or free[0].uuid != gpu.uuid:
            return False

        settings = resolved_settings(self.config, job)
        logs = settings.output_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        attempt = self.attempts[job.slug] + 1
        stdout_path = logs / f"attempt_{attempt:02d}.stdout.log"
        stderr_path = logs / f"attempt_{attempt:02d}.stderr.log"
        stdout_handle = stdout_path.open("ab", buffering=0)
        stderr_handle = stderr_path.open("ab", buffering=0)
        command = [
            str(self.python),
            "-u",
            "-m",
            MODULE,
            "--config",
            str(self.config),
            "--task",
            job.task,
            "--method",
            job.method,
            "--seed",
            str(job.seed),
            "--resume",
        ]
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
        self.attempts[job.slug] = attempt
        active = ActiveJob(
            key=job,
            gpu_index=gpu.index,
            gpu_uuid=gpu.uuid,
            pid=process.pid,
            attempt=attempt,
            started_at_unix=time.time(),
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )
        self.active[job.slug] = active
        _write_json_atomic(
            settings.output_dir / "process.json",
            {
                "status": "running",
                "task": job.task,
                "method": job.method,
                "seed": job.seed,
                "attempt": attempt,
                "pid": process.pid,
                "manager_pid": os.getpid(),
                "hostname": socket.gethostname(),
                "gpu_index": gpu.index,
                "gpu_uuid": gpu.uuid,
                "cuda_device_order": "PCI_BUS_ID",
                "cuda_visible_devices": str(gpu.index),
                "command": command,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "started_at_unix": active.started_at_unix,
            },
        )
        print(
            f"[P12 queue] launched {job.slug} pid={process.pid} gpu={gpu.index} "
            f"attempt={attempt}",
            flush=True,
        )
        return True

    def run(self) -> int:
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def interrupt_manager(_signum, _frame):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt_manager)
        try:
            with QueueLock(self.lock_path):
                devices, _ = query_gpu_inventory(nvidia_smi=self.nvidia_smi)
                missing = [index for index in self.gpu_indices if index not in devices]
                if missing:
                    raise RuntimeError(f"Requested GPU indices do not exist: {missing}")
                print(
                    f"[P12 queue] {len(self.jobs)} jobs, GPUs={self.gpu_indices}, "
                    f"seeds={self.seeds}, adaptation_seeds={self.adaptation_seeds}; "
                    f"state={self.state_path}",
                    flush=True,
                )
                try:
                    while True:
                        self._poll_active()
                        state = self._persist()
                        if state["counts"]["complete"] == len(self.jobs):
                            self._persist(note="all jobs complete")
                            return 0
                        if not self.active and not self._next_ready_job():
                            # A failed common start leaves its three adaptations
                            # blocked. Exit nonzero instead of reporting completion.
                            self._persist(note="terminal failures or blocked dependencies")
                            return 2

                        devices, applications = query_gpu_inventory(
                            nvidia_smi=self.nvidia_smi
                        )
                        free = free_requested_gpus(
                            self.gpu_indices,
                            devices,
                            applications,
                            reserved_indices=(
                                active.gpu_index for active in self.active.values()
                            ),
                        )
                        for gpu in free:
                            job = self._next_ready_job()
                            if job is None:
                                break
                            self._launch(job, gpu)
                        self._persist()
                        time.sleep(self.poll_seconds)
                except KeyboardInterrupt:
                    self._persist(note="manager interrupted; child jobs left running")
                    print(
                        "[P12 queue] manager interrupted; training children remain alive "
                        "and a restarted queue will recover their PIDs",
                        flush=True,
                    )
                    return 130
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dependency-aware, compute-safe multi-GPU P12 scheduler."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_int_list, default=(1, 2, 3, 4, 5))
    parser.add_argument("--seeds", type=parse_int_list, default=(2026, 2027, 2028))
    parser.add_argument(
        "--adaptation-seeds",
        type=parse_int_list,
        help="subset receiving BP/FA adaptation (default: every --seeds value)",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--status", action="store_true", help="print the persisted status and exit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queue = MultiGPUQueue(
        config=args.config,
        gpu_indices=args.gpus,
        seeds=args.seeds,
        adaptation_seeds=args.adaptation_seeds,
        python=args.python,
        repo_root=args.repo_root,
        poll_seconds=args.poll_seconds,
        max_retries=args.max_retries,
        nvidia_smi=args.nvidia_smi,
    )
    if args.status:
        state = _read_json(queue.state_path)
        if state is None:
            print(f"No queue state exists at {queue.state_path}")
            return 1
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
