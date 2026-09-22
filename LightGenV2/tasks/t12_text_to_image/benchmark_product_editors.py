"""Matched end-to-end latency benchmark for Qwen electronic and optical editors."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from .compact_product_model import prepare_compact_optical_unet
from .electronic_turbo_infer import _load_adapter
from .feature_cache import _qwen_prompts
from .half_qwen import load_half_qwen_text_encoder
from .product_repair_model import RepairModelConfig, expand_reference_conditioning, one_step_edit
from .product_scene_replace_data import ProductBackgroundReplacementDataset


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, round((len(ordered)-1)*fraction))]


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean_ms": statistics.mean(values), "p50_ms": _percentile(values,.5), "p95_ms": _percentile(values,.95)}


@torch.inference_mode()
def _benchmark_variant(
    *, name: str, qwen_layers: int, optical_variant: bool, checkpoint: Path,
    initial_unet: Path, turbo_checkpoint: Path, adapter_checkpoint: Path,
    qwen_checkpoint: Path, data_dir: Path, instruction_cache: Path,
    device: torch.device, warmup: int, repeats: int,
) -> dict[str, Any]:
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
    model_config=RepairModelConfig(**payload["model_config"]);training=payload["training_config"]
    qwen,processor,qwen_report=load_half_qwen_text_encoder(qwen_checkpoint,device,keep_layers=qwen_layers)
    adapter,_=_load_adapter(adapter_checkpoint,device);adapter.load_state_dict(payload["adapter"]);adapter.eval()
    unet=UNet2DConditionModel.from_pretrained(initial_unet,subfolder="unet",variant="fp16",torch_dtype=torch.float32,local_files_only=True).to(device)
    expand_reference_conditioning(unet)
    optical=None
    if optical_variant:
        optical,_=prepare_compact_optical_unet(unet,model_config);optical.hardware_bypass=True
    unet.load_state_dict(payload["unet"]);unet.eval()
    if optical is not None: optical.hardware_bypass=True
    dtype=torch.float16
    vae=AutoencoderKL.from_pretrained(turbo_checkpoint,subfolder="vae",variant="fp16",torch_dtype=dtype,local_files_only=True).to(device).eval()
    scheduler=EulerDiscreteScheduler.from_pretrained(turbo_checkpoint,subfolder="scheduler",local_files_only=True);scheduler.set_timesteps(1,device=device);sigma=scheduler.sigmas[0]
    raw=ProductBackgroundReplacementDataset(data_dir,"test",256,instruction_cache)[0]
    image=raw["reference"][None].to(device=device,dtype=dtype);mask=raw["foreground_mask"][None].to(device=device,dtype=dtype);foreground=raw["foreground_rgb"][None].to(device=device,dtype=dtype)
    prompt=raw["prompt"]
    token_inputs={key:value.to(device) for key,value in _qwen_prompts(processor,[prompt]).items()}
    noise=torch.randn((1,4,32,32),device=device,dtype=torch.float32,generator=torch.Generator(device=device).manual_seed(42))

    def run()->None:
        # Match the mixed-precision path used by the training and inference
        # entrypoints.  Without autocast the float32 UNet weights force this
        # benchmark onto an unrealistically slow FP32 execution path.
        with torch.autocast(device_type=device.type,dtype=dtype):
            encoded=vae.encode(image).latent_dist.mode()*vae.config.scaling_factor
            outputs=qwen(**token_inputs,output_hidden_states=False,return_dict=True,use_cache=False)
            hidden=outputs.last_hidden_state.float();attention=token_inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled=(hidden*attention).sum(1)/attention.sum(1).clamp_min(1)
            condition=adapter.condition(pooled)
            latent=one_step_edit(unet,noise,encoded.float(),condition,sigma,residual_scale=float(training["residual_scale"]),noise_scale=float(training["noise_scale"]))
            rgb=vae.decode(latent.to(dtype)/vae.config.scaling_factor,return_dict=False)[0]
            _=rgb*(1-mask)+foreground*mask

    for _ in range(warmup): run()
    torch.cuda.synchronize(device)
    values=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();run();end.record();torch.cuda.synchronize(device);values.append(float(start.elapsed_time(end)))
    measured=_summary(values)
    physical=measured["mean_ms"]+(1.0447*6 if optical_variant else 0.0)
    result={
        "name":name,"qwen_layers":qwen_layers,"software_optics_executed":False if optical_variant else None,
        "measured_optical_bypass":measured,"physical_optical_latency_ms":1.0447*6 if optical_variant else 0.0,
        "estimated_hardware_mean_ms":physical,"prompt":prompt,
        "qwen_counted_parameters":qwen_report["counted_text_encoder_parameters"],
        "token_embedding_parameters_excluded":qwen_report["token_embedding_parameters_excluded_by_project_convention"],
    }
    del qwen,processor,adapter,unet,vae,payload
    torch.cuda.empty_cache()
    return result


def main()->int:
    p=argparse.ArgumentParser()
    for name in ("baseline-checkpoint","optical-checkpoint","initial-unet","turbo-checkpoint","adapter-checkpoint","qwen-checkpoint","data-dir","baseline-instruction-cache","optical-instruction-cache","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--warmup",type=int,default=30);p.add_argument("--repeats",type=int,default=100);p.add_argument("--device",default="cuda")
    a=p.parse_args();device=torch.device(a.device)
    baseline=_benchmark_variant(name="full_qwen_electronic_baseline",qwen_layers=28,optical_variant=False,checkpoint=a.baseline_checkpoint.resolve(),initial_unet=a.initial_unet.resolve(),turbo_checkpoint=a.turbo_checkpoint.resolve(),adapter_checkpoint=a.adapter_checkpoint.resolve(),qwen_checkpoint=a.qwen_checkpoint.resolve(),data_dir=a.data_dir.resolve(),instruction_cache=a.baseline_instruction_cache.resolve(),device=device,warmup=a.warmup,repeats=a.repeats)
    optical=_benchmark_variant(name="three_layer_qwen_compact_optical",qwen_layers=3,optical_variant=True,checkpoint=a.optical_checkpoint.resolve(),initial_unet=a.initial_unet.resolve(),turbo_checkpoint=a.turbo_checkpoint.resolve(),adapter_checkpoint=a.adapter_checkpoint.resolve(),qwen_checkpoint=a.qwen_checkpoint.resolve(),data_dir=a.data_dir.resolve(),instruction_cache=a.optical_instruction_cache.resolve(),device=device,warmup=a.warmup,repeats=a.repeats)
    report={"schema_version":1,"batch_size":1,"image_size":256,"tokenization_and_model_loading_excluded":True,"baseline":baseline,"optical":optical,"speedup_x":baseline["estimated_hardware_mean_ms"]/optical["estimated_hardware_mean_ms"],"latency_reduction_fraction":1-optical["estimated_hardware_mean_ms"]/baseline["estimated_hardware_mean_ms"]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
