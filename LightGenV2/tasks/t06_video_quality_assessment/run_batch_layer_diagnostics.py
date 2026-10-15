"""One bounded CPU/GPU diagnosis suite; never waits on or occupies a busy GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as original

TASK = Path(__file__).resolve().parent
ORIGINAL = TASK/"runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02"
CHILD_TIMEOUT_SECONDS = 600


def child_command(name, phase, output):
    report = json.loads((ORIGINAL/f"{name}_batch01/report.json").read_text())
    env = original.QWEN_ENV if name=="qwen" else reproduction.ENV_PATHS[name]
    diag = "diagnostic_qwen" if name=="qwen" else "diagnostic_deepseek_interfacefix"
    command = [env,"-u","-m",__package__+".diagnose_batch_layers","--phase",phase,"--baseline",name,
        "--manifest",report["manifest"],"--checkpoint",report["checkpoint"],"--model",report["model"],
        "--output",str(output),"--previous-diagnostic",str(ORIGINAL/diag)]
    if name=="deepseek":
        command += ["--reproduction-receipt",str(original.PREVIOUS/name/"reproduction_receipt.json")]
    return command


def run(args):
    import fcntl
    baselines = tuple(getattr(args,"baselines",("qwen","deepseek")))
    if not baselines or len(set(baselines)) != len(baselines) or set(baselines)-{"qwen","deepseek"}:
        raise ValueError("Unique supported baselines required")
    if not args.output.is_relative_to((TASK/"runs/simulation").resolve()):
        raise ValueError("Canonical task run required")
    if args.phase=="cpu":
        if args.output.exists(): raise FileExistsError("Refuse CPU suite overwrite")
        args.output.mkdir(parents=True,exist_ok=False)
    else:
        cpu = json.loads((args.output/"suite_cpu.json").read_text())
        if cpu["status"]!="complete": raise ValueError("CPU identity/position audit not complete")
        if list(baselines) != cpu.get("baselines",["qwen","deepseek"]):
            raise ValueError("Baseline selection/order differs from CPU receipt")
        if any((args.output/f"gpu_{name}").exists() for name in ("qwen","deepseek")):
            raise FileExistsError("Preserve previous GPU attempt; do not retry blindly")
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=frozen.A100_UUID:
            raise ValueError("Wrong physical GPU")
        reproduction.cuda_guard()  # Return rather than create a GPU context or wait.
    sources = [Path(__file__),TASK/"diagnose_batch_layers.py",Path(frozen.__file__),TASK/"diagnose_batch_numerics.py"]
    sha = {f.name:frozen.sha256(f) for f in sources}
    if args.phase=="gpu" and sha != cpu["source_sha256"]:
        raise ValueError("Source changed since CPU audit; requires explicit new provenance")
    with (TASK/"runs/simulation/.batch_layer_diagnostics.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = dict(status="running",pid=os.getpid(),phase=args.phase,completed=[],source_sha256=sha,
            git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            samples=8,batches=[1,2,4],baselines=list(baselines),child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
            limits="No training, full558 re-evaluation, cost replacement, lab access or foreign process interference")
        destination = args.output/f"suite_{args.phase}.json"
        frozen.write_json(destination,state)
        try:
            for name in baselines:
                if sha != {f.name:frozen.sha256(f) for f in sources}: raise ValueError("Running source changed")
                if args.phase=="gpu": reproduction.cuda_guard()
                state["current"]=name; frozen.write_json(destination,state)
                output = args.output/f"{args.phase}_{name}"
                command = child_command(name,args.phase,output)
                frozen.write_json(args.output/f"{args.phase}_{name}.command.json",command)
                env = dict(os.environ,CUDA_VISIBLE_DEVICES="-1" if args.phase=="cpu" else frozen.A100_UUID)
                with (args.output/f"{args.phase}_{name}.log").open("x") as stream:
                    subprocess.run(command,check=True,env=env,stdout=stream,stderr=subprocess.STDOUT,
                        timeout=CHILD_TIMEOUT_SECONDS)
                child = json.loads((output/"status.json").read_text())
                if child["status"]!="complete": raise ValueError("Incomplete child status")
                state["completed"].append(name); frozen.write_json(destination,state)
            state["status"]="complete"
        except BaseException as error:
            state.update(status="failed",error=repr(error)); raise
        finally:
            frozen.write_json(destination,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("cpu","gpu"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baselines",nargs="+",choices=("qwen","deepseek"),default=["qwen","deepseek"])
    args=parser.parse_args(); args.output=args.output.resolve(); run(args)


if __name__=="__main__": main()
