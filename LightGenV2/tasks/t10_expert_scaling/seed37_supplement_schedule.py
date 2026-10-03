"""Complete seed 37 for the Kather expert-count study on packed GPUs.

The four N=4 MoE runs are reused from the fixed-478 study because their
checkpoint configuration is identical.  This scheduler trains the remaining
12 MoE runs and one D2NN baseline for each expert count (16 new trainings).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


GRID = {16: [1, 4, 8, 16], 25: [1, 6, 12, 25], 49: [1, 12, 24, 49]}


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reuse-n4", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--slots-per-gpu", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "logs_seed37").mkdir(exist_ok=True)

    reused = []
    for k in (1, 2, 3, 4):
        name = f"moe_oeo_N4_k{k}_L4_lr0.002_s37"
        source = a.reuse_n4 / f"kather2016__moe_oeo_N4_k{k}_L4_lr0.002_s37"
        destination = a.out / name
        if not (source / "result.json").exists():
            raise FileNotFoundError(source / "result.json")
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=True)
        reused.append(name)

    jobs = []
    # Longest runs first so the makespan is governed by useful work, not tail latency.
    for n in (49, 25, 16):
        for k in GRID[n]:
            jobs.append(dict(arch="moe_oeo", experts=n, top_k=k,
                             name=f"moe_oeo_N{n}_k{k}_L4_lr0.002_s37"))
        jobs.append(dict(arch="d2nn_expert_global", experts=n, top_k=n,
                         name=f"d2nn_expert_global_N{n}_k{n}_L4_lr0.002_s37"))
    jobs.append(dict(arch="d2nn_expert_global", experts=4, top_k=4,
                     name="d2nn_expert_global_N4_k4_L4_lr0.002_s37"))
    for j in jobs:
        j.update(layers=4, lr=.002, seed=37)

    pending = [j for j in jobs if not (a.out / j["name"] / "result.json").exists()]
    complete = [j for j in jobs if j not in pending]
    running = {}
    failed = []

    def status(state="training"):
        save(a.out / "seed37_status.json", dict(
            state=state, expected_new_trainings=len(jobs), reused_n4_moe=reused,
            complete=len(complete), remaining=len(pending),
            running={f"gpu{g}_slot{s}": j for (g, s), (_, _, j) in running.items()},
            failed=failed, gpus=a.gpus, slots_per_gpu=a.slots_per_gpu))

    try:
        while pending or running:
            for gpu in a.gpus:
                # Do not collide with a process that was already using this physical GPU.
                # Once this scheduler owns a slot, the second packed slot may start normally.
                if not any(g == gpu for g, _ in running):
                    used = int(subprocess.check_output([
                        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], text=True).strip())
                    if used > 200:
                        continue
                for slot in range(a.slots_per_gpu):
                    key = (gpu, slot)
                    if key in running or not pending:
                        continue
                    job = pending.pop(0)
                    uuid = subprocess.check_output([
                        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                        "--format=csv,noheader"], text=True).strip()
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
                    cmd = [sys.executable, "-u", "-m", "LightGenV2.tasks.t10_expert_scaling.train",
                           "--data", str(a.data), "--out", str(a.out / job["name"]),
                           "--arch", job["arch"], "--experts", str(job["experts"]),
                           "--top-k", str(job["top_k"]), "--layers", "4", "--lr", ".002",
                           "--microbatch", "2", "--epochs", str(a.epochs), "--seed", "37",
                           "--resume"]
                    log = (a.out / "logs_seed37" / f"{job['name']}.log").open("a")
                    proc = subprocess.Popen(cmd, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    running[key] = (proc, log, job)
            for key, (proc, log, job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[key]
                if proc.returncode:
                    failed.append(dict(job=job, exit_code=proc.returncode))
                    raise RuntimeError(f"Job failed: {job}")
                complete.append(job)
            status()
            if pending or running:
                time.sleep(10)
        status("complete")
    except BaseException as error:
        status("failed_or_interrupted")
        save(a.out / "seed37_error.json", dict(error=repr(error), failed=failed))
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
        save(a.out / "seed37_release_check.json", dict(
            own_children_exited=True, failed=failed, timestamp=time.time()))


if __name__ == "__main__":
    main()
