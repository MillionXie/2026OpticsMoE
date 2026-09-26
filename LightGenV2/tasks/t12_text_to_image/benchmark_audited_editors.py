"""Fixed-weight Qwen28 baseline comparison; measured CUDA vs hardware proxies."""
import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .audited_unified import architecture_report
from .benchmark_three_task_bundle import _timed
from .electronic_turbo_infer import _load_adapter
from .feature_cache import _qwen_prompts
from .half_qwen import load_half_qwen_text_encoder
from .product_repair_model import expand_reference_conditioning, one_step_edit
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from .qwen_mini_small import PromptEmbeddingLookup
from .sealed_editor import build_sealed
from .small_fullframe import _edge


class CachedRoute(nn.Module):
    def __init__(self, routing):
        super().__init__()
        self.routing = routing

    def forward(self, _):
        return self.routing


class IdentityBranch(nn.Module):
    def forward(self, value, *args):
        return value


def bypass_physics(model):
    spatial = model.editor.bottleneck if model.kind == "small" else model.unet.mid_block.hybrid
    for optical in (model.text.optical, spatial.optical):
        # Keep electronic amplitude encoding/reload and CCD readout. Replace
        # only optical router propagation and expert/global detector simulation.
        optical.core.router = CachedRoute({k: v.detach() for k, v in optical.core.last_routing.items()})
        optical.measured_expert_ccd = optical.last_raw_expert_ccd.detach()
        optical.measured_global_ccd = optical.last_raw_ccd.detach()


def prepare_text_boundary(model, embeddings, mask):
    valid = mask.bool()
    packed = embeddings[:, valid[0]]
    projected = model.text.input_projection(packed).detach()
    valid = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
    padding = ~valid

    def hidden(self, *_):
        e1 = self.layers[0](projected, valid)
        with torch.autocast(projected.device.type, enabled=False):
            o1, routing, lengths = self.optical.run_expert_block(projected.float(), padding)
        stage2 = self.fusion1(e1, o1.to(e1.dtype)).masked_fill(padding[..., None], 0)
        e2 = self.layers[1](stage2, valid)
        with torch.autocast(projected.device.type, enabled=False):
            field = self.optical.encode_global_input(stage2.float(), padding, routing)
            o2 = self.optical.run_global_block(field, lengths, padding, torch.float32)
        value = self.norm(self.fusion2(e2, o2.to(e2.dtype)))
        return value[:, -1]

    model.text.hidden = types.MethodType(hidden, model.text)
    return int(valid.sum())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ("assets", "qwen", "large", "small", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    assets = args.assets
    dataset = ExpandedUnifiedProductEditDataset(assets/"datasets/abo_cleanrender_lamp_table_pillow_256_v1", "test", 256,
                                               assets/"datasets/abo_unified_expanded_instructions_qwen2_v2.pt")
    row = dataset[0]
    ref = row["reference"].unsqueeze(0).to(device)
    lookup = PromptEmbeddingLookup(assets/"datasets/abo_unified_expanded_qwen_embeddings_v2.pt")
    embeddings, mask, _ = lookup.batch([row["prompt"]], device)
    result = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": sys.argv, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "batch_size": 1, "resolution": 256, "warmup": args.warmup, "repeats": args.repeats,
              "prompt": row["prompt"], "sample_id": row["sample_id"],
              "boundary": "first language block input -> RGB; embedding/projection/packing/loading/transfer excluded",
              "precision": "CUDA autocast fp16; audited optics internals fp32",
              "hardware_note": "Cached route and CCD replace FFT only for timing, never quality. Parallel branch removal is an optimistic lower-bound proxy, not real hardware timing."}
    for name in ("large", "small"):
        checkpoint = getattr(args, name)
        model = build_sealed(torch.load(checkpoint, map_location="cpu", weights_only=False)).to(device).eval()
        report = architecture_report(model)
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = prepare_text_boundary(model, embeddings.float(), mask)
        shape = (1, 4, 32, 32) if name == "large" else ref.shape
        noise = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(1042))

        def run():
            with torch.autocast("cuda", dtype=torch.float16):
                return model(ref, embeddings.float(), mask, noise)

        simulation = _timed(run, device, args.warmup, args.repeats)
        bypass_physics(model)
        electronic = _timed(run, device, args.warmup, args.repeats)
        model.text.layers = nn.ModuleList([IdentityBranch(), IdentityBranch()])
        spatial = model.editor.bottleneck if name == "small" else model.unet.mid_block.hybrid
        spatial.electronic1 = IdentityBranch()
        spatial.electronic2 = IdentityBranch()
        lower = _timed(run, device, args.warmup, args.repeats)
        result[name] = {"checkpoint": str(checkpoint), "sha256": sha(checkpoint),
                        "counted_parameters": report["parameters_plus_fixed_condition_values"], "prompt_tokens": tokens,
                        "simulation": simulation, "fft_bypassed_electronic_retained": electronic,
                        "fft_and_parallel_branches_bypassed": lower,
                        "requested_six_layer_proxy_ms": lower["mean_ms"] + 6.2682,
                        "two_sequential_paths_proxy_ms": lower["mean_ms"] + 12.5364,
                        "electronic_retained_plus_two_paths_ms": electronic["mean_ms"] + 12.5364}
        print(json.dumps({name: result[name]}), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()

    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    baseline_path = assets/"runs/abo_scene_replace_28layer_electronic_baseline_v1/best_model.pt"
    payload = torch.load(baseline_path, map_location="cpu", weights_only=False, mmap=True)
    qwen, processor, qreport = load_half_qwen_text_encoder(args.qwen, device, keep_layers=28)
    adapter, _ = _load_adapter(assets/"runs/677deac6/qwen_sd_turbo_one_step_text_aug_seed42/best_adapter.pt", device)
    adapter.load_state_dict(payload["adapter"]); adapter.eval()
    unet = UNet2DConditionModel.from_pretrained(assets/"models/bk-sdm-v2-tiny", subfolder="unet", variant="fp16",
                                               torch_dtype=torch.float32, local_files_only=True)
    expand_reference_conditioning(unet)
    unet.load_state_dict(payload["unet"]); unet = unet.to(device).eval()
    vae = AutoencoderKL.from_pretrained(assets/"models/sd-turbo-fp16", subfolder="vae", variant="fp16",
                                       torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(assets/"models/sd-turbo-fp16", subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1, device=device)

    def tokens_for(prompts):
        values = {k: v[:, -64:].to(device) for k, v in _qwen_prompts(processor, prompts).items()}
        return qwen.embed_tokens(values.pop("input_ids")).detach(), values

    def pooled_for(emb, inputs):
        hidden = qwen(inputs_embeds=emb, **inputs, use_cache=False, return_dict=True).last_hidden_state.float()
        valid = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden*valid).sum(1)/valid.sum(1).clamp_min(1)

    def image_for(reference, pooled, noise):
        latent = vae.encode(reference.half()).latent_dist.mode()*vae.config.scaling_factor
        edited = one_step_edit(unet, noise, latent.float(), adapter.condition(pooled.float()), scheduler.sigmas[0],
                               residual_scale=float(payload["training_config"]["residual_scale"]),
                               noise_scale=float(payload["training_config"]["noise_scale"]))
        return vae.decode(edited.half()/vae.config.scaling_factor, return_dict=False)[0]

    emb, inputs = tokens_for([row["prompt"]])
    noise = torch.randn((1, 4, 32, 32), device=device)
    def run_baseline():
        with torch.autocast("cuda", dtype=torch.float16):
            pooled = pooled_for(emb, inputs)
            return image_for(ref, pooled, noise)
    measured = _timed(run_baseline, device, args.warmup, args.repeats)
    counted = qreport["counted_text_encoder_parameters"] + sum(p.numel() for m in (adapter, unet, vae) for p in m.parameters())
    result["baseline"] = {"checkpoint": str(baseline_path), "sha256": sha(baseline_path),
                          "counted_parameters": counted, "qwen_report": qreport, "measured": measured,
                          "training_scope": "historical background-only lamp checkpoint; object/joint and new categories are transfer diagnostics, NOT a retrained matched-task baseline"}
    for name in ("large", "small"):
        entry = result[name]
        for mode in ("requested_six_layer_proxy_ms", "two_sequential_paths_proxy_ms", "electronic_retained_plus_two_paths_ms"):
            entry[mode+"_speedup"] = measured["mean_ms"]/entry[mode]
    (args.output/"timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"baseline": result["baseline"]}), flush=True)

    from .evaluate_unified_editor import TorchvisionInceptionFeatures
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    fid = FrechetInceptionDistance(feature=TorchvisionInceptionFeatures().to(device)).to(device)
    kid = KernelInceptionDistance(feature=TorchvisionInceptionFeatures().to(device), subsets=10, subset_size=100).to(device)
    conditions = {}
    metrics = {}
    torch.manual_seed(42)
    generator = torch.Generator(device=device).manual_seed(1042)
    for batch_index, batch in enumerate(DataLoader(dataset, batch_size=2, num_workers=2)):
        with torch.autocast("cuda", dtype=torch.float16):
            for prompt in batch["prompt"]:
                if prompt not in conditions:
                    e, t = tokens_for([prompt])
                    conditions[prompt] = pooled_for(e, t)
            pooled = torch.cat([conditions[p] for p in batch["prompt"]])
            reference, target = batch["reference"].to(device), batch["target"].to(device)
            noise = torch.randn((len(reference), 4, 32, 32), device=device, generator=generator)
            output = image_for(reference, pooled, noise).float()
        for i, mode in enumerate(batch["mode"]):
            values = {"mse": float(F.mse_loss(output[i], target[i])),
                      "edge_l1": float(F.l1_loss(_edge(output[i:i+1]), _edge(target[i:i+1])))}
            for key in ("overall", mode, batch["category"][i]+":"+mode):
                group = metrics.setdefault(key, {"count": 0, "mse": 0., "edge_l1": 0.})
                group["count"] += 1
                for k, v in values.items(): group[k] += v
        real = target.add(1).mul(127.5).clamp(0,255).byte()
        fake = output.add(1).mul(127.5).clamp(0,255).byte()
        fid.update(real, real=True); fid.update(fake, real=False)
        kid.update(real, real=True); kid.update(fake, real=False)
        if batch_index % 100 == 0: print("quality batches", batch_index, flush=True)
    for group in metrics.values():
        group["mse"] /= group["count"]; group["edge_l1"] /= group["count"]
        group["psnr_db"] = 10*math.log10(4/max(group["mse"], 1e-12))
    mean, std = kid.compute()
    quality = {"metrics": metrics, "samples": len(dataset), "postprocessing": "raw full decoded RGB, no GT masking/pasting",
               "distribution_metrics": {"fid_imagenet_inception_variant": float(fid.compute()), "kid_mean": float(mean), "kid_std": float(std)},
               "caveat": result["baseline"]["training_scope"], "timing_provenance": result["git_commit"]}
    (args.output/"baseline_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality), flush=True)


if __name__ == "__main__":
    main()
