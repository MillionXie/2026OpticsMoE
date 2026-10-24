"""Actual-checkpoint, actual-input LightGen neural-stage measurement.

This first adapter binds the requested ABO .8225 checkpoint. No optimizer,
synthetic detector inputs, model surgery, warm-up or fastest-sample selection.
The original forward (including its untimed numerical optics) produces scores.
Only declared neural spans plus original input processing enter the hardware
projection. The complete software wall and every excluded span remain raw data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("lightgen_stage_meter", HERE / "stage_meter.py")
meter_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meter_lib)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gpu_snapshot(uuid):
    command = ["nvidia-smi", f"--id={uuid}", "--query-gpu=uuid,name,power.limit,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    gpu = subprocess.check_output(command, text=True).strip()
    processes = []
    result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    for line in result.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0] == uuid:
            processes.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "name": parts[2], "memory_mib": parts[3]})
    return {"monotonic": time.perf_counter(), "gpu": gpu, "processes": processes}


def guard(uuid):
    result = gpu_snapshot(uuid)
    if not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise RuntimeError("Explicit private MPS pipe required, not only a nominal100 environment variable")
    def control(command):
        r = subprocess.run(["nvidia-cuda-mps-control"], input=command + "\n", capture_output=True, text=True, timeout=5)
        if r.returncode:
            raise RuntimeError("MPS control failed: " + r.stderr)
        return {int(s) for s in re.findall(r"^\s*(\d+)\s*$", r.stdout, re.M)}
    servers = control("get_server_list")
    clients = set().union(*(control("get_client_list " + str(s)) for s in servers)) if servers else set()
    self_ids = {os.getpid()}
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("NSpid:"):
            self_ids.update(map(int, line.split()[1:]))
    foreign = [p for p in result["processes"] if p["pid"] not in self_ids | servers]
    if clients - self_ids:
        raise RuntimeError(f"Foreign MPS clients: {clients - self_ids}")
    if foreign:
        raise RuntimeError(f"Foreign GPU compute: {foreign}")
    result.update(mps_servers=sorted(servers), mps_clients=sorted(clients))
    return result


def patch_abo(model, meter):
    """Explicit accounting ledger, with source lines for every selected span."""
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone import optics
    meter.wrap(optics, "bounded_amplitude", "slm.bounded_tanh_nonlinearity", family="amplitude_encoding_nn")
    meter.wrap(model.frontend, "patches", "frontend.patch_position", family="frontend")
    meter.wrap(model.frontend, "embed", "frontend.token_embedding", family="frontend")
    meter.wrap(model.frontend, "merge", "bridge.native_merger", family="bridge")
    for mode in ("vision", "language"):
        core = getattr(model, mode)
        body = meter_lib.statements(core, "forward")
        index = meter_lib.assign_index(body, "latent")
        meter.patch_statements(core, "forward", [(f"{mode}.input_adapter_norm", "adapter", True, index, index)])
        for name in ("output_norm", "output_adapter"):
            if hasattr(core, name):
                meter.wrap(getattr(core, name), "forward", f"{mode}.{name}", family="adapter")
        path = core.optics
        body = meter_lib.statements(path, "encode")
        projected, rms = meter_lib.assign_index(body, "projected"), meter_lib.assign_index(body, "rms")
        meter.patch_statements(path, "encode", [
            (f"{mode}.encode_linear_norm_softplus", "amplitude_encoding_nn", True, projected, projected),
            (f"{mode}.encode_rms", "amplitude_encoding_nn", True, rms, len(body) - 1),
        ])
        body = meter_lib.statements(path.router, "forward")
        first = meter_lib.assign_index(body, "energy")
        last = meter_lib.assign_index(body, "weights")
        meter.patch_statements(path.router, "forward", [(f"{mode}.router_energy_top2", "router", True, first, last)])
        body = meter_lib.statements(path, "decode")
        value, relative = meter_lib.assign_index(body, "value"), meter_lib.assign_index(body, "relative")
        pooled, readout = meter_lib.assign_index(body, "pooled"), meter_lib.assign_index(body, "readout")
        meter.patch_statements(path, "decode", [
            (f"{mode}.ccd_normalization", "ccd_nonlinearity", True, value, relative),
            (f"{mode}.ccd_log_pool_layernorm_relu", "ccd_readout", True, pooled, readout),
            (f"{mode}.ccd_output_linear", "ccd_readout", True, len(body) - 1, len(body) - 1),
        ])
        for index, residual in enumerate(core.blocks):
            meter.wrap(residual, "forward", f"{mode}.parallel_residual{index + 1}", family="parallel_residual", charged=False)
        meter.wrap(path.router.propagator, "forward", f"{mode}.router_fft", family="simulation_only", charged=False)
        meter.wrap(path.propagator, "forward", f"{mode}.expert_global_fft", family="simulation_only", charged=False)
        meter.wrap(path, "fanout", f"{mode}.field_layout", family="layout", charged=False)
    meter.wrap(model.readout, "forward", "head.spatial2x2_retrieval", family="head")
    if model.late_rgb is not None:
        raise ValueError("Requested .8225 is not the later rank72 RGB-bypass model")


def prepare_abo_input(processor, image, device):
    """Original processor arguments; audit metadata comes from CPU tensors.

    Never execute a GPU reduction or D2H merely to count tokens between calls.
    The four transferred tensors and the original template are unchanged.
    """
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import template
    cpu = processor(text=[template(processor, image)], images=[image], padding=True, return_tensors="pt")
    metadata = {"input_ids_shape": list(cpu["input_ids"].shape),
                "pixel_values_shape": list(cpu["pixel_values"].shape),
                "image_grid_thw": cpu["image_grid_thw"].tolist(),
                "valid_input_tokens": int(cpu["attention_mask"].sum()),
                "processed_image_size": list(image.size)}
    return {k: cpu[k].to(device) for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}, metadata


def load_abo(args, device):
    import torch
    from torch.nn import functional as F
    from transformers import AutoProcessor
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval

    expected = "0b1ff5b682b833f8c4fd25e6330147e993e974272d7d2812fa679082ef0e87bc"
    if meter_lib.file_sha256(args.checkpoint) != expected:
        raise ValueError("ABO .8225 checkpoint SHA mismatch")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = OpticalRetrieval(payload["metadata"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.assets / "processor"), local_files_only=True)
    protocol = json.loads(args.manifest.read_text())
    rows = protocol["rows"]
    gallery_rows = [r for r in rows if r["split"] in ("train", "gallery")]
    queries = [r for r in rows if r["split"] == "query"]
    if (len(gallery_rows), len(queries)) != (1600, 800):
        raise ValueError("ABO enrolled gallery1600/query800 contract required")
    features_path = args.checkpoint.parent / "features.pt"
    features = torch.load(features_path, map_location="cpu", weights_only=True)
    by_id = {key: i for i, key in enumerate(features["ids"])}
    gallery = F.normalize(torch.stack([features["vectors"][by_id[r["sample_id"]]] for r in gallery_rows]).float(), dim=-1).to(device)
    info = {"checkpoint_sha256": expected, "checkpoint": str(args.checkpoint), "metadata": payload["metadata"],
            "manifest": str(args.manifest), "manifest_sha256": meter_lib.file_sha256(args.manifest),
            "offline_gallery_cache": str(features_path), "offline_gallery_cache_sha256": meter_lib.file_sha256(features_path),
            "gallery_count": len(gallery_rows), "full_test_count": len(queries), "target_reference_hit1": .8225,
            "selected_metric": payload.get("metrics"), "audit": model.audit()}
    return model, processor, queries[:args.samples], gallery_rows, gallery, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["abo_image_to_image"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--repeat", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--gpu-uuid", default="GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.samples != 200:
        raise ValueError("Formal measurement is200 actual queries; use separate test fixtures for smoke")
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing measurement")
    import torch
    if not args.inspect_only:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
            raise ValueError("Bind the full A100 UUID explicitly before CUDA initialization")
        if os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") != "100":
            raise ValueError("This table requires MPS100")
        admission = guard(args.gpu_uuid)
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA A100-PCIE-40GB" or properties.multi_processor_count != 108:
            raise ValueError("Wrong GPU or unexpected client-visible SM cap")
    else:
        admission = None
        device = torch.device("cpu")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    model, processor, queries, gallery_rows, gallery, info = load_abo(args, device)
    meter = meter_lib.StageMeter(None if args.inspect_only else torch.cuda)
    patch_abo(model, meter)
    info.update({"protocol_id": "lightgen_qwen_input_inclusive_200x3_20261024_v1",
                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                 "script_sha256": meter_lib.file_sha256(__file__), "stage_meter_sha256": meter_lib.file_sha256(HERE / "stage_meter.py"),
                 "stage_ledger": meter.ledger, "repeat": args.repeat, "samples": len(queries), "explicit_warmup": 0,
                 "scope": "input-inclusive stage-accounted hardware projection; wall primary; excluded layout/residual/FFT retained",
                 "mps_percent": 100, "gpu_admission": admission, "inspection_only": args.inspect_only})
    info["cpu_threads"] = torch.get_num_threads()
    args.output.mkdir(parents=True)
    write_json(args.output / "identity.json", info)
    if args.inspect_only:
        meter.close()
        from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import picture, inputs
        query = queries[0]
        batch = inputs(processor, [picture(args.data / query["image_path"], model.metadata.get("input_preprocessing", "center_crop"))], device)
        with torch.inference_mode():
            original = model(batch)
            patch_abo(model, meter)
            instrumented = model(batch)
        difference = float((original - instrumented).abs().max())
        if not torch.equal(original, instrumented):
            raise RuntimeError(f"Instrumentation changes the original CPU forward: {difference}")
        check = {"status": "cpu_inspection_passed", "modules": len(meter.ledger) // 2,
                 "queries": len(queries), "checkpoint_sha256": info["checkpoint_sha256"],
                 "real_sample_id": query["sample_id"], "bitwise_forward_equivalence": True, "max_abs_difference": difference,
                 "scope": "one actual-image CPU arithmetic contract, not formal GPU timing or accuracy"}
        write_json(args.output / "cpu_contract.json", check)
        print(json.dumps(check))
        meter.close()
        return
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import picture, inputs
    samples, processes = [], [admission]
    first_start, last_return = None, None
    # Same independent one-second MPS process audit as the coordinated baseline.
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as mps_engine
    from types import SimpleNamespace
    mps_engine.GPU = args.gpu_uuid
    mps_engine.PIPE = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
    process_guard = mps_engine.MPSGuard(SimpleNamespace(deadline=time.time() + 3600))
    try:
        process_guard.bind()
        with torch.inference_mode():
            for i, query in enumerate(queries):
                process_guard.fast()
                meter.sample_index, meter.valid_samples = i, 1
                start = time.perf_counter()
                if first_start is None:
                    first_start = start
                gap = 0. if last_return is None else start - last_return
                with meter.scope("input.file_decode_processor_h2d", "input", gpu=False):
                    path = args.data / query["image_path"]
                    image = picture(path, model.metadata.get("input_preprocessing", "center_crop"))
                    batch, input_metadata = prepare_abo_input(processor, image, device)
                    torch.cuda.synchronize()
                call_start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vector = model(batch)
                with meter.scope("head.gallery_similarity_stable_ranking", "search"):
                    scores = vector.float() @ gallery.T
                    ranking = scores.argsort(dim=1, descending=True, stable=True)
                with meter.scope("output.cpu_return", "output", gpu=False):
                    cpu_ranking = ranking.cpu()
                    predicted_index = int(cpu_ranking[0, 0])
                    predicted_id = gallery_rows[predicted_index]["product_id"]
                last_return = time.perf_counter()
                samples.append({"index": i, "sample_id": query["sample_id"], "image_path": query["image_path"],
                                "expected_product_id": query["product_id"], "predicted_product_id": predicted_id,
                                "hit1": int(predicted_id == query["product_id"]), "input_start": start,
                                "ranking": cpu_ranking[0].tolist(),
                                "model_call_start": call_start, "cpu_result_return": last_return,
                                "request_wall_ms": (last_return - start) * 1000, "internal_gap_ms": gap * 1000,
                                **input_metadata})
        processes.append(guard(args.gpu_uuid))
        # Validate original image hashes after, not as a pre-read that warms IO.
        for r, query in zip(samples, queries):
            digest = meter_lib.file_sha256(args.data / query["image_path"])
            r["image_sha256"] = digest
            from PIL import Image
            with Image.open(args.data / query["image_path"]) as original_image:
                r["original_image_size"] = list(original_image.size)
            if query.get("image_sha256") and query["image_sha256"] != digest:
                raise ValueError("Input image changed relative to the sealed manifest")
        for entry in meter.ledger:
            if meter_lib.file_sha256(entry["source_file"]) != entry["source_sha256"]:
                raise RuntimeError("A measured module source changed during this run")
        charged_ms = sum(r["synchronized_wall_ms"] for r in meter.rows if r["charged"]) + sum(r["internal_gap_ms"] for r in samples)
        opt_seconds = len(samples) * 6 * .0010447
        energy = meter_lib.optical_energy_j(opt_seconds, charged_ms / 1000)
        paired = [r for r in meter.rows if r["charged"] and r["cuda_event_ms"] is not None]
        paired_wall = sum(r["synchronized_wall_ms"] for r in paired)
        paired_events = sum(r["cuda_event_ms"] for r in paired)
        summary = {"status": "complete", "repeat": args.repeat, "valid_samples": len(samples),
                   "measured_subset_hit1": statistics.fmean(r["hit1"] for r in samples), "full_test_reference_hit1": .8225,
                   "software_simulation_full_window_ms": (last_return - first_start) * 1000,
                   "charged_electronic_ms_total": charged_ms, "physical_passes": len(samples) * 6,
                   "optical_ms_total": opt_seconds * 1000, "phase_accounted_energy_j_total": energy,
                   "cost": meter_lib.equivalent_cost(opt_seconds + charged_ms / 1000, energy, len(samples)),
                   "primary": "synchronized wall NN intervals and input/output plus internal dispatch gaps; no layout/residual/FFT",
                   "power_basis": "user-fixed phase powers, not software-FFT telemetry: optical121.746W/electronic104.230W",
                   "common_token_basis": "pending matching Qwen input audit; LightGen native token count is diagnostic only",
                   "paired_neural_timers": {"wall_ms_total": paired_wall, "event_ms_total": paired_events,
                                            "wall_minus_event_ms_total": paired_wall-paired_events,
                                            "wall_event_ratio": paired_wall/paired_events if paired_events else None},
                   "op_count_status": "pending separate matching untimed replay; never reuse old synthetic-shape OP"}
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"type": type(exc).__name__, "error": str(exc), "completed_samples": len(samples)})
        raise
    finally:
        process_guard.close(args.output / "process_audit.json")
        meter.close()
        write_csv(args.output / "samples.csv", samples)
        write_csv(args.output / "stages.csv", meter.rows)
        write_json(args.output / "gpu_processes.json", processes)
        write_json(args.output / "stage_ledger.json", meter.ledger)
        write_json(args.output / "files_manifest.json", [{"file": p.name, "bytes": p.stat().st_size, "sha256": meter_lib.file_sha256(p)} for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "files_manifest.json"])


if __name__ == "__main__":
    main()
