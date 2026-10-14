"""Visible, low-memory A100 reservation explicitly authorized by the user.

No busy loop kernels, model load, inference or measurement while others run.
This is a cooperative claim, not a GPU-exclusive scheduler or eviction tool.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction

RESERVATION_BYTES = 64 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024


def foreign_processes(records, own_pid):
    return [row for row in records
            if row["gpu_uuid"] == frozen.A100_UUID and row["pid"] != str(own_pid)]


def processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return [dict(zip(("pid", "gpu_uuid", "process"), map(str.strip, fields)))
            for fields in csv.reader(result.stdout.splitlines()) if len(fields) == 3]


def run(args):
    import torch
    import fcntl
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Reservation must bind only the approved A100 UUID")
    reproduction.require_new_output(args.output)
    if (args.status_file.parent != args.output.parent or args.status_file.exists()
            or args.deadline_unix <= time.time()
            or args.deadline_unix > time.time() + 10800):
        raise ValueError("New sidecar status and a bounded <=3-hour deadline required")
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w, validate_cuda_device
    if gpu_power_limit_w(frozen.A100_UUID) != 250:
        raise RuntimeError("Power limit changed; do not alter another user's device settings")
    before = processes()
    with (args.output.parent / ".a100_reservation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Initialize one visible context, allocate only 64 MiB. Never fill VRAM
        # or run compute merely to force another process to leave.
        gpu_name = validate_cuda_device("A100")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < MIN_FREE_BYTES + RESERVATION_BYTES:
            raise RuntimeError("Insufficient spare VRAM for safe small reservation")
        buffer = torch.empty(RESERVATION_BYTES, dtype=torch.uint8, device="cuda:0")
        identity = {
            "status": "visible_reservation_waiting_not_measuring", "pid": os.getpid(),
            "gpu_uuid": frozen.A100_UUID, "gpu_name": gpu_name,
            "reserved_tensor_bytes": RESERVATION_BYTES,
            "note": "CUDA context overhead is additional; no model, kernels or dummy load; cooperative visibility, not exclusive ownership",
            "gpu_free_bytes_before_allocation": free_bytes, "gpu_total_bytes": total_bytes,
            "before_reservation_processes": before, "started_unix": time.time(),
            "deadline_unix": args.deadline_unix, "measurement_output": str(args.output),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "reservation_source_sha256": frozen.sha256(Path(__file__)),
            "excluded_from_results": "All reservation/waiting activity and the interrupted initial round are excluded from this full remeasurement",
        }
        empty_since = None
        try:
            while time.time() < args.deadline_unix:
                all_records = processes()
                foreign = foreign_processes(all_records, os.getpid())
                if not foreign:
                    empty_since = empty_since or time.monotonic()
                else:
                    empty_since = None
                identity.update(last_check_unix=time.time(), current_foreign_processes=foreign,
                                current_processes=all_records)
                frozen.write_json(args.status_file, identity)
                print(f"reservation PID={os.getpid()} foreign_A100_pids={[row['pid'] for row in foreign]} measuring=False", flush=True)
                if empty_since is not None and time.monotonic() - empty_since >= 5:
                    if gpu_power_limit_w(frozen.A100_UUID) != 250:
                        raise RuntimeError("Power limit changed while waiting")
                    # exec destroys this CUDA context before the fresh suite
                    # initializes anything; it is not whitelisted in any audit.
                    identity.update(status="releasing_context_then_starting_full_remeasurement", released_unix=time.time())
                    frozen.write_json(args.status_file, identity)
                    del buffer
                    torch.cuda.empty_cache()
                    command = [sys.executable, "-u", "-m", __package__ + ".run_temporal_telemetry_suite",
                               "--manifest", str(args.manifest), "--output", str(args.output)]
                    os.execv(sys.executable, command)
                time.sleep(5)
            identity.update(status="reservation_wait_budget_exhausted", stopped_unix=time.time())
            frozen.write_json(args.status_file, identity)
        except BaseException as error:
            identity.update(status="reservation_failed", error=repr(error), stopped_unix=time.time())
            frozen.write_json(args.status_file, identity)
            raise
        finally:
            # No occupied foreign process is terminated or reconfigured.
            if "buffer" in locals():
                del buffer
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "status-file"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    args.manifest, args.output, args.status_file = args.manifest.resolve(), args.output.resolve(), args.status_file.resolve()
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Manifest identity mismatch")
    run(args)


if __name__ == "__main__":
    main()
