"""Three-dataset fixed-478 study with a validation-range pilot gate."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


TOP_K = [1, 2, 3, 4]
SEEDS = [17, 27, 37]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def job(dataset, data, arch, seed, top_k):
    name = f"{dataset}__{arch}_N4_k{top_k}_L4_lr0.002_s{seed}"
    return dict(dataset=dataset, data=str(data), arch=arch, experts=4,
                top_k=top_k, layers=4, lr=.002, seed=seed, name=name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kather", type=Path, required=True)
    p.add_argument("--blood", type=Path, required=True)
    p.add_argument("--organc", type=Path, required=True)
    p.add_argument("--reuse-kather", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    if len(a.gpus) != 4 or len(set(a.gpus)) != 4:
        raise ValueError("Exactly four distinct GPUs are required")
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "logs").mkdir()
    datasets = {"kather2016": a.kather, "bloodmnist": a.blood, "organcmnist": a.organc}
    moe = [job(ds, path, "moe_oeo", seed, k)
           for ds, path in datasets.items() for seed in SEEDS for k in TOP_K]
    baseline = [job(ds, path, "d2nn_same_aperture", seed, 4)
                for ds, path in datasets.items() for seed in SEEDS]
    if len(moe) != 36 or len(baseline) != 9:
        raise AssertionError((len(moe), len(baseline)))

    cached = []
    if a.reuse_kather:
        for j in moe:
            if j["dataset"] != "kather2016" or j["seed"] not in [17, 27]:
                continue
            source = a.reuse_kather / j["name"].split("__", 1)[1]
            destination = a.out / j["name"]
            if (source / "result.json").exists():
                destination.symlink_to(source, target_is_directory=True)
                cached.append(j)

    # New datasets must pass the requested 70--92% range before multi-seed expansion.
    pilots = [j for j in moe if j["dataset"] in ["bloodmnist", "organcmnist"] and j["seed"] == 17]
    rest = [j for j in moe + baseline if j not in cached and j not in pilots]
    pending = list(pilots)
    running = {}
    finished = list(cached)
    failed = []
    stage = "pilot"
    save(a.out / "design.json", dict(
        datasets=list(datasets), license="CC BY 4.0", active_panel_side_px=478,
        canvas_side_px=518, experts=4, top_k=TOP_K, seeds=SEEDS,
        moe_runs=36, d2nn_runs=9, total_runs=45, cached_runs=len(cached),
        baseline="d2nn_same_aperture", layers=4, epochs=a.epochs,
        performance_gate=[.70, .92], test_read=False))
    save(a.out / "jobs.json", dict(moe=moe, baseline=baseline))

    def status(state="training"):
        save(a.out / "status.json", dict(
            state=state, stage=stage, total=45, complete=len(finished),
            cached=len(cached), remaining=len(pending) + (len(rest) if stage == "pilot" else 0),
            running={str(g): j for g, (_, _, j) in running.items()}, failed=failed))

    try:
        while pending or running or stage == "pilot":
            for gpu in a.gpus:
                if gpu in running or not pending:
                    continue
                memory = int(subprocess.check_output([
                    "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True).strip())
                if memory > 200:
                    continue
                j = pending.pop(0)
                folder = a.out / j["name"]
                uuid = subprocess.check_output([
                    "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                    "--format=csv,noheader"], text=True).strip()
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
                cmd = [sys.executable, "-u", "-m", "LightGenV2.tasks.t10_expert_scaling.train",
                       "--data", j["data"], "--out", str(folder), "--arch", j["arch"],
                       "--experts", "4", "--top-k", str(j["top_k"]), "--layers", "4",
                       "--lr", ".002", "--microbatch", "2", "--epochs", str(a.epochs),
                       "--seed", str(j["seed"])]
                log = (a.out / "logs" / f"{j['name']}.log").open("w")
                proc = subprocess.Popen(cmd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                running[gpu] = (proc, log, j)
                save(a.out / f"gpu{gpu}.json", dict(pid=proc.pid, gpu_uuid=uuid, job=j, command=cmd))
            for gpu, (proc, log, j) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close(); del running[gpu]
                if proc.returncode:
                    failed.append(dict(job=j, exit_code=proc.returncode))
                    raise RuntimeError(f"Job failed: {j}")
                finished.append(j)
            if stage == "pilot" and not pending and not running:
                gate = {}
                for ds in ["bloodmnist", "organcmnist"]:
                    vals = []
                    for j in pilots:
                        if j["dataset"] == ds:
                            result = json.loads((a.out / j["name"] / "result.json").read_text())
                            vals.append(result["val"]["accuracy"])
                    gate[ds] = dict(values=vals, best=max(vals), passed=.70 <= max(vals) <= .92)
                save(a.out / "pilot_gate.json", gate)
                if not all(x["passed"] for x in gate.values()):
                    stage = "replacement_required"; status("replacement_required"); return
                stage = "full"; pending.extend(rest); rest.clear()
            status()
            if pending or running:
                time.sleep(10)
        status("complete")
    except BaseException as error:
        status("failed_or_interrupted")
        save(a.out / "error.json", dict(error=repr(error)))
        raise
    finally:
        for proc, log, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            log.close()
        snapshot = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"], text=True)
        save(a.out / "release_check.json", dict(
            own_children_exited=True, remaining_gpu_processes=snapshot, failed=failed))


if __name__ == "__main__":
    main()
