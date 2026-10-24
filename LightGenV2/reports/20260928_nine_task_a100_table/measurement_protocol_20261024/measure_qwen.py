"""Frozen, original-head baselines, continuous file-to-CPU-result200x3.

No predecoded/GPU-resident test inputs, inference warmup or new readout training.
The first adapter is the exact enrolled-SKU .85125 Qwen baseline, not the
different optical .8225 checkpoint. More tasks require explicit adapters.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("ours_meter_runtime", HERE / "measure_ours.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sha = shared.meter_lib.file_sha256
write = shared.write_json


def load_abo(args, device):
    import torch
    from torch.nn import functional as F
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    manifest = json.loads(args.manifest.read_text())
    if sha(args.manifest) != "f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde":
        raise ValueError("Wrong enrolled SKU protocol")
    rows = manifest["rows"]
    gallery_rows = [r for r in rows if r["split"] in ("train", "gallery")]
    queries = [r for r in rows if r["split"] == "query"]
    if (len(gallery_rows), len(queries)) != (1600, 800):
        raise ValueError("Require original1600 gallery/800 TEST queries")
    feature_path = args.reference.parent / "normal_features.pt"
    reference = json.loads(args.reference.read_text())
    if reference.get("model_kind") != "qwen64" or abs(reference["metrics"]["normal"]["hit_at_1"] - .85125) > 1e-10:
        raise ValueError("Not the requested .85125 frozen native-aspect Qwen baseline")
    if reference["manifest_sha256"] != sha(args.manifest):
        raise ValueError("Reference uses another data split")
    cache = torch.load(feature_path, map_location="cpu", weights_only=True)
    by_id = {v:i for i,v in enumerate(cache["ids"])}
    if cache["vectors"].shape != (2400, 64):
        raise ValueError("Original frozen Qwen64 gallery required")
    gallery = F.normalize(torch.stack([cache["vectors"][by_id[r["sample_id"]]] for r in gallery_rows]).float(), dim=-1).to(device)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, min_pixels=50176, max_pixels=50176)
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    assets = {p.name:sha(p) for p in sorted(args.model.iterdir())
              if p.is_file() and (p.suffix in (".json", ".safetensors") or p.name.endswith(".model"))}
    if not any(p.endswith(".safetensors") for p in assets):
        raise ValueError("Backbone tensor identity missing")
    if assets["config.json"] != reference["model_config_sha256"]:
        raise ValueError("Original .85125 backbone configuration differs")
    identity = dict(model_assets=assets, reference=str(args.reference), reference_sha256=sha(args.reference),
                    gallery_cache=str(feature_path), gallery_cache_sha256=sha(feature_path),
                    full_test_reference_hit1=.85125, full_test_count=800, gallery_count=1600,
                    precision="native BF16 frozen Qwen, original64-prefix normalized descriptor",
                    prompt="original standalone.io.template, no prompt or architecture modification")
    return model, processor, queries[:200], gallery_rows, gallery, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["abo_image_to_image"], required=True)
    for name in ("model", "manifest", "data", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repeat", type=int, choices=(1,2,3), required=True)
    p.add_argument("--gpu-uuid", default="GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or
            os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") != "100"):
        raise ValueError("Full approved UUID plus private MPS100 required before CUDA init")
    admission = shared.guard(args.gpu_uuid)
    import torch
    from torch.nn import functional as F
    from PIL import Image, ImageOps
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import template
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if props.name != "NVIDIA A100-PCIE-40GB" or props.multi_processor_count != 108:
        raise ValueError("Wrong GPU/MPS resources")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    model, processor, queries, gallery_rows, gallery, identity = load_abo(args, device)
    telemetry_root = REPO/"LightGenV2/tasks/t06_video_quality_assessment"
    bound_sources = [Path(__file__), HERE/"measure_ours.py", HERE/"stage_meter.py",
                     *[telemetry_root/name for name in ("e2e_telemetry_observer.py", "nvml_observer_process.py",
                                                       "batch_telemetry.py", "strict_tf_window_20261020.py")]]
    identity.update(protocol_id="lightgen_qwen_input_inclusive_200x3_20261024_v1", task=args.task,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    sources={str(f.relative_to(REPO)):sha(f) for f in bound_sources},
                    manifest_sha256=sha(args.manifest), gpu_uuid=args.gpu_uuid, mps_percent=100, repeat=args.repeat,
                    first_sample_retained=True, explicit_warmup_forwards=0, command=sys.argv,
                    GPU_admission=admission, torch=torch.__version__, cuda=torch.version.cuda,
                    cpu_threads=torch.get_num_threads(),
                    timer="first raw file read through200th CPU ranking; all199 gaps retained")
    args.output.mkdir(parents=True)
    write(args.output/"identity.json", identity)
    calls, outputs, telemetry = [], [], []
    engine.GPU, engine.PIPE = args.gpu_uuid, os.environ["CUDA_MPS_PIPE_DIRECTORY"]
    guard = engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800))
    raw_telemetry = args.output/"telemetry_stdout.jsonl"
    sampler = ProcessTelemetrySampler(args.gpu_uuid, raw_telemetry, command=observer_command(args.gpu_uuid,raw_telemetry))
    try:
        guard.bind()
        sampler.start()
        sampler.set_phase("continuous_file_to_cpu_no_warmup")
        time.sleep(.15)  # observer bracketing only, no model inference
        with torch.inference_mode():
            for i,row in enumerate(queries):
                guard.fast(); sampler.check()
                start = time.perf_counter()
                with Image.open(args.data/row["image_path"]) as original:
                    original_size = list(original.size)
                    image = ImageOps.exif_transpose(original).convert("RGB")
                cpu = processor(text=[template(processor,image)], images=[image], padding=True, return_tensors="pt")
                metadata = dict(input_ids_shape=list(cpu["input_ids"].shape), pixel_values_shape=list(cpu["pixel_values"].shape),
                                image_grid_thw=cpu["image_grid_thw"].tolist(),
                                valid_input_tokens=int(cpu["attention_mask"].sum()), original_size=original_size,
                                input_ids=cpu["input_ids"].tolist(), attention_mask=cpu["attention_mask"].tolist())
                prepared = time.perf_counter()
                inputs = {k:cpu[k].to(device) for k in ("input_ids","attention_mask","pixel_values","image_grid_thw")}
                torch.cuda.synchronize()
                model_start = time.perf_counter()
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    hidden = model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
                    mask = inputs["attention_mask"].bool()
                    positions = torch.arange(mask.shape[1],device=device)[None].expand_as(mask).masked_fill(~mask,-1).amax(1)
                    vector = F.normalize(hidden[torch.arange(len(mask),device=device),positions,:64].float(),dim=-1)
                # Match the original FP32 cosine ranking after descriptor
                # extraction; do not silently autocast retrieval scores to BF16.
                scores = vector.float() @ gallery.T
                ranking = scores.argsort(dim=1,descending=True,stable=True)
                cpu_ranking = ranking.cpu()
                predicted = gallery_rows[int(cpu_ranking[0,0])]["product_id"]
                returned = time.perf_counter()
                calls.append(dict(index=i,sample_id=row["sample_id"],image_path=row["image_path"],
                    expected_product_id=row["product_id"],predicted_product_id=predicted,hit1=int(predicted==row["product_id"]),
                    input_start=start,prepare_end=prepared,model_start=model_start,cpu_result_return=returned,**metadata))
                outputs.append(cpu_ranking[0].tolist())
        sampler.request("barrier");time.sleep(.12)
        telemetry = sampler.stop();sampler=None
        start,end = calls[0]["input_start"],calls[-1]["cpu_result_return"]
        elapsed = end-start
        if len(calls)!=200 or len({r["sample_id"] for r in calls})!=200:
            raise ValueError("Wrong test call denominator")
        energy = interval_energy([(start,end)],telemetry,338.2)
        for call,row,ranking in zip(calls,queries,outputs):
            call["image_sha256"] = sha(args.data/row["image_path"])
            call["ranking"] = ranking
            if row["image_sha256"]!=call["image_sha256"]:
                raise ValueError("Input content differs from sealed protocol")
        for source,digest in identity["sources"].items():
            if sha(REPO/source)!=digest:raise ValueError("Bound measurement source changed")
        guard.check()
        report = dict(status="complete",**identity,measured_samples=200,
            measured_subset_hit1=sum(r["hit1"] for r in calls)/200,
            continuous_wall_s=elapsed,total_system_energy_j=energy["energy_j"],power_integration=energy,
            mean_board_power_w=(energy["energy_j"]-338.2*elapsed)/elapsed,
            common_reference_tokens=sum(r["valid_input_tokens"] for r in calls),
            cost=shared.meter_lib.equivalent_cost(elapsed,energy["energy_j"],200))
        write(args.output/"report.json",report)
        print(json.dumps({k:report[k] for k in ("status","measured_samples","measured_subset_hit1","continuous_wall_s","mean_board_power_w","cost")}))
    except BaseException as error:
        write(args.output/"failure.json",dict(error=repr(error),completed_samples=len(calls)))
        raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        guard.close(args.output/"process_audit.json")
        if telemetry:save_samples(args.output/"telemetry.csv",telemetry)
        shared.write_csv(args.output/"calls.csv",calls)
        write(args.output/"files_manifest.json",[{"file":f.name,"sha256":sha(f),"bytes":f.stat().st_size}
                for f in sorted(args.output.iterdir()) if f.is_file() and f.name!="files_manifest.json"])


if __name__=="__main__":main()
