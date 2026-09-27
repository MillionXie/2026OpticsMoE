"""Owned-process supervisor: rebuild missing caches, four single-GPU runs, report.

Never kills other users' processes. Each child exits before its GPU is released.
Only start this script following explicit user authorization of four GPUs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import GROUPS, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Exactly four explicitly authorized physical GPU indices")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--soft-targets", type=Path, required=True)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not gpu.isdigit() for gpu in gpus):
        parser.error("Require four distinct numeric GPU indices")
    root = ROOT / "runs/simulation" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    for path in (args.dataset_root, args.qwen_model, args.manifest, args.soft_targets):
        if not path.exists():
            raise ValueError(f"Required rebuild asset missing: {path}")
    state = {"status": "preparing", "supervisor_pid": os.getpid(), "gpus": gpus, "children": [],
             "started": time.time(), "command": sys.argv,
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    report = root / "supervisor.json"
    owned = []
    gpu_uuids = {}
    write_json(report, state)

    def idle():
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
        used = {a.strip(): int(memory.strip()) for line in raw.splitlines() for a,uuid,memory in [line.split(",")]}
        gpu_uuids.update({a.strip(): uuid.strip() for line in raw.splitlines() for a,uuid,memory in [line.split(",")]})
        if any(gpu not in used or used[gpu] > 512 for gpu in gpus):
            raise RuntimeError(f"Requested GPUs no longer idle; do not terminate others: {used}")

    def start(command, gpu, log):
        stream = log.open("w", encoding="utf-8")
        # CUDA's default FASTEST_FIRST order can differ from nvidia-smi indices.
        # Bind the verified PHYSICAL UUID, never an ambiguous numeric ordinal.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuids[gpu], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        owned.append((child, stream))
        state["children"].append({"pid": child.pid, "gpu": gpu, "gpu_uuid": gpu_uuids[gpu], "command": command, "log": str(log)})
        write_json(report, state)
        return child, stream

    def finish(child, stream):
        code = child.wait()
        stream.close()
        for item in state["children"]:
            if item["pid"] == child.pid:
                item.update(exit_code=code, exited_at=time.time())
        write_json(report, state)
        return code

    try:
        idle()
        assets = ROOT / "assets/cache/rebuilt_4f_20260927"
        assets.mkdir(parents=True, exist_ok=True)
        vision, language = assets / "vision_49x1024_quality14.pt", assets / "language_temporal_2048.pt"
        paths = {"dataset_root": str(args.dataset_root), "manifest": str(args.manifest),
                 "vision_cache": str(vision), "language_cache": str(language),
                 "training_soft_targets": str(args.soft_targets)}
        write_json(root / "paths.json", paths)
        command = [sys.executable, "-I", str(ROOT / "prepare_cache.py"),
                   "--dataset-root", str(args.dataset_root), "--model-path", str(args.qwen_model),
                   "--manifest", str(args.manifest), "--vision-output", str(vision),
                   "--language-output", str(language), "--target", "temporal", "--frame-count", "4",
                   "--token-grid", "7", "--batch-size", "2", "--chunk-rows", "16", "--device", "cuda"]
        child, stream = start(command, gpus[0], root / "cache_rebuild.log")
        if finish(child, stream):
            raise RuntimeError("Cache rebuild failed; see cache_rebuild.log")
        write_json(root / "assets_sha256.json", {k: sha256(Path(v)) for k,v in paths.items() if k != "dataset_root"})
        # Strict backend validates cache identity before ANY formal training starts.
        sys.path.insert(0, str(ROOT / "runtime"))
        from study import make_config, split_for_selection
        import yaml
        from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import load_settings
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import load_single_metric_cache
        raw = make_config("r0_post", paths=paths)
        config = root / "asset_check.yaml"
        config.write_text(yaml.safe_dump(raw), encoding="utf-8")
        payload = load_single_metric_cache(load_settings(config))
        _, split = split_for_selection(payload)
        write_json(root / "selection_split.json", split)
        del payload
        idle()
        state["status"] = "training"
        children = []
        for group, gpu in zip(GROUPS, gpus):
            command = [sys.executable, "-I", str(ROOT / "run.py"), "--group", group, "--phase", "train",
                       "--paths", str(root / "paths.json"), "--device", "cuda", "--allow-uncalibrated-noise",
                       "--output", str(root / group)]
            child, stream = start(command, gpu, root / f"{group}.log")
            children.append((group, gpu, child, stream))
        results = {}
        for group, gpu, child, stream in children:
            code = finish(child, stream)
            results[group] = {"training_exit_code": code, "metrics": {}}
            if code:
                continue
            checkpoint = root / group / "best_checkpoint.pt"
            for split_name in ("train", "validation", "test"):
                destination = root / group / f"final_{split_name}"
                command = [sys.executable, "-I", str(ROOT / "run.py"), "--group", group, "--phase", "evaluate",
                           "--paths", str(root / "paths.json"), "--checkpoint", str(checkpoint), "--eval-split", split_name,
                           "--device", "cuda", "--noise-scale", "1", "--output", str(destination)]
                evaluation, log = start(command, gpu, root / f"{group}_{split_name}.log")
                if finish(evaluation, log):
                    results[group]["evaluation_failed"] = split_name
                    break
                results[group]["metrics"][split_name] = json.loads((destination / "evaluation.json").read_text())["metrics"]
            results[group]["checkpoint_sha256"] = sha256(checkpoint)
        state["status"] = "complete" if all(len(r["metrics"]) == 3 for r in results.values()) else "failed_or_partial"
        write_json(root / "comparison.json", {"status": state["status"], "groups": results,
                   "metric_scope": "simulation common 8um/DC30/noise1; not optical hardware measurements"})
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        for child, stream in owned:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            stream.close()
        try:
            processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
            gpu_pids = {line.strip() for line in processes.splitlines()}
            state["owned_gpu_pids_remaining"] = [child.pid for child, _ in owned if str(child.pid) in gpu_pids]
        except Exception as error:
            state["gpu_release_check_error"] = repr(error)
        state["finished"] = time.time()
        write_json(report, state)


if __name__ == "__main__":
    main()
