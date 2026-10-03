"""Four-GPU Kather top-k scan: four expert counts x four k values x two seeds.

Completed runs from a prior corrected-profile directory may be reused by symlink.
Two N=49 D2NN repetitions are scheduled first as a reproducibility check.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


GRID = {4: [1, 2, 3, 4], 16: [1, 4, 8, 16], 25: [1, 6, 12, 25], 49: [1, 12, 24, 49]}


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reuse", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    if len(a.gpus) != 4 or len(set(a.gpus)) != 4:
        raise ValueError("This schedule requires four distinct GPUs")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "logs").mkdir(exist_ok=True)

    scan = []
    for seed in [17, 27]:
        for n, ks in GRID.items():
            for k in ks:
                name = f"moe_oeo_N{n}_k{k}_L4_lr0.002_s{seed}"
                scan.append(dict(arch="moe_oeo", experts=n, top_k=k, layers=4,
                                 lr=.002, seed=seed, name=name, kind="topk_scan"))
    if len(scan) != 32:
        raise AssertionError(len(scan))

    cached = []
    pending_scan = []
    for job in scan:
        destination = a.out / job["name"]
        source = a.reuse / job["name"] if a.reuse else None
        if destination.exists() and (destination / "result.json").exists():
            cached.append(job)
        elif source and (source / "result.json").exists():
            destination.symlink_to(source, target_is_directory=True)
            cached.append(job)
        else:
            pending_scan.append(job)

    rechecks = []
    for seed in [17, 27]:
        name = f"d2nn_expert_global_N49_k49_L4_lr0.002_s{seed}_recheck"
        rechecks.append(dict(arch="d2nn_expert_global", experts=49, top_k=49,
                             layers=4, lr=.002, seed=seed, name=name,
                             kind="d2nn_n49_recheck"))
    pending = rechecks + pending_scan
    save(a.out / "scan_design.json", dict(
        protocol="corrected_dynamic_router_full_aperture_v1",
        expert_counts=list(GRID), top_k_grid=GRID, seeds=[17, 27],
        total_scan_runs=32, cached_scan_runs=len(cached),
        pending_scan_runs=len(pending_scan), extra_d2nn_n49_rechecks=2,
        test_read=False))
    save(a.out / "jobs.json", dict(scan=scan, rechecks=rechecks))
    save(a.out / "identity.json", dict(pid=os.getpid(), command=sys.argv,
         gpus=a.gpus, source_commit=subprocess.check_output(
             ["git", "rev-parse", "HEAD"], text=True).strip()))

    running = {}
    finished = list(cached)
    failed = []

    def write_status(state="training"):
        save(a.out / "status.json", dict(
            state=state, scan_total=32,
            scan_complete=sum(j["kind"] == "topk_scan" for j in finished),
            cached_scan_runs=len(cached),
            rechecks_complete=sum(j["kind"] == "d2nn_n49_recheck" for j in finished),
            remaining=len(pending),
            running={str(g): j for g, (_, _, j) in running.items()}, failed=failed))

    try:
        while pending or running:
            for gpu in a.gpus:
                if gpu in running or not pending:
                    continue
                memory = int(subprocess.check_output([
                    "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True).strip())
                if memory > 200:
                    continue
                job = pending.pop(0)
                folder = a.out / job["name"]
                uuid = subprocess.check_output([
                    "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                    "--format=csv,noheader"], text=True).strip()
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
                cmd = [sys.executable, "-u", "-m", "LightGenV2.tasks.t10_expert_scaling.train",
                       "--data", str(a.data), "--out", str(folder),
                       "--arch", job["arch"], "--experts", str(job["experts"]),
                       "--top-k", str(job["top_k"]), "--layers", "4", "--lr", ".002",
                       "--microbatch", "2", "--epochs", str(a.epochs), "--seed", str(job["seed"])]
                log = (a.out / "logs" / f"{job['name']}.log").open("w")
                proc = subprocess.Popen(cmd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                running[gpu] = (proc, log, job)
                save(a.out / f"gpu{gpu}.json", dict(pid=proc.pid, gpu_uuid=uuid,
                                                     job=job, command=cmd))
            for gpu, (proc, log, job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[gpu]
                if proc.returncode:
                    failed.append(dict(job=job, exit_code=proc.returncode))
                    raise RuntimeError(f"Job failed: {job}")
                finished.append(job)
            write_status()
            if pending or running:
                time.sleep(10)
        write_status("complete")
    except BaseException as error:
        write_status("failed_or_interrupted")
        save(a.out / "error.json", dict(error=repr(error)))
        raise
    finally:
        for proc, log, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        snapshot = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv"], text=True)
        save(a.out / "release_check.json", dict(
            own_children_exited=True, remaining_gpu_processes=snapshot, failed=failed))


if __name__ == "__main__":
    main()
