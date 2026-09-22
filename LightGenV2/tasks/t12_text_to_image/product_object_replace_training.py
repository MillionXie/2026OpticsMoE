"""Cache and train the compact one-pass text-guided ABO object replacer."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .compact_product_model import prepare_compact_optical_unet
from .compact_turbo import latent_gradient_loss
from .electronic_turbo_infer import _load_adapter
from .product_object_replace_data import TARGET_CATALOGUE_SIZE, ProductObjectReplacementDataset
from .product_repair_model import RepairModelConfig, architecture_report, expand_reference_conditioning, one_step_edit
from .product_scene_training import SceneTrainingConfig, _masked_l1, _paired_noise, _seed_everything, load_scene_config


class ObjectLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def __len__(self) -> int:
        return len(self.payload["reference"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = {
            key: self.payload[key][index].float()
            for key in ("reference", "target", "edit_mask", "preserve_mask", "qwen_text")
        }
        for key in ("sample_ids", "prompts", "target_categories", "target_catalogue_indices"):
            result[key] = self.payload[key][index]
        return result


@torch.inference_mode()
def cache_object_latents(
    *, data_dir: Path, instruction_cache: Path, vae_checkpoint: Path,
    output_dir: Path, image_size: int, device: torch.device,
    batch_size: int = 16, num_workers: int = 4,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKL

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        vae_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype, local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    scale = float(vae.config.scaling_factor)
    summary = {"schema_version": 1, "task": "lamp -> text-selected object", "splits": {}}
    for split in ("train", "val", "test"):
        dataset = ProductObjectReplacementDataset(data_dir, split, image_size, instruction_cache)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
        payload: dict[str, list[Any]] = {key: [] for key in (
            "reference", "target", "edit_mask", "preserve_mask", "qwen_text",
            "sample_ids", "prompts", "target_categories", "target_catalogue_indices",
        )}
        for batch in loader:
            reference = batch["reference"].to(device=device, dtype=dtype, non_blocking=True)
            target = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
            reference_latent = vae.encode(reference).latent_dist.mode() * scale
            target_latent = vae.encode(target).latent_dist.mode() * scale
            edit = F.interpolate(batch["edit_mask"].float(), reference_latent.shape[-2:], mode="area")
            payload["reference"].append(reference_latent.half().cpu())
            payload["target"].append(target_latent.half().cpu())
            payload["edit_mask"].append(edit.half().cpu())
            payload["preserve_mask"].append((1.0-edit).half().cpu())
            payload["qwen_text"].append(batch["qwen_text"].bfloat16().cpu())
            for destination, source in (
                ("sample_ids", "sample_id"), ("prompts", "prompt"),
                ("target_categories", "target_category"),
            ):
                payload[destination].extend(batch[source])
            payload["target_catalogue_indices"].extend(batch["target_catalogue_index"].tolist())
        packed = {
            key: torch.cat(value) if key in {"reference", "target", "edit_mask", "preserve_mask", "qwen_text"} else value
            for key, value in payload.items()
        }
        torch.save(packed, output_dir / f"{split}.pt")
        summary["splits"][split] = len(dataset)
    (output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    del vae
    if device.type == "cuda": torch.cuda.empty_cache()
    return summary


class TextObjectRouter(nn.Module):
    def __init__(self, text_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, TARGET_CATALOGUE_SIZE))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())


@torch.inference_mode()
def evaluate_object_model(unet, adapter, router, loader, sigma, device, *, residual_scale: float, noise_scale: float) -> dict[str, float]:
    unet.eval(); adapter.eval(); router.eval()
    totals = {"latent_mse": 0.0, "edit_l1": 0.0, "preserve_change_l1": 0.0, "reference_mse": 0.0, "catalogue_accuracy": 0.0}
    samples = 0
    generator = torch.Generator(device=device).manual_seed(8721)
    for batch in loader:
        reference = batch["reference"].to(device); target = batch["target"].to(device)
        edit = batch["edit_mask"].to(device); preserve = batch["preserve_mask"].to(device)
        text = batch["qwen_text"].to(device); labels = torch.as_tensor(batch["target_catalogue_indices"], device=device)
        condition = adapter.condition(text); noise = _paired_noise(reference, generator)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = one_step_edit(unet, noise, reference, condition, sigma, residual_scale=residual_scale, noise_scale=noise_scale).float()
        count = len(reference)
        totals["latent_mse"] += float(F.mse_loss(output, target))*count
        totals["reference_mse"] += float(F.mse_loss(reference, target))*count
        totals["edit_l1"] += float(_masked_l1(output, target, edit))*count
        totals["preserve_change_l1"] += float(_masked_l1(output, reference, preserve))*count
        totals["catalogue_accuracy"] += float((router(text).argmax(-1)==labels).float().mean())*count
        samples += count
    result = {key:value/samples for key,value in totals.items()}
    result["mse_improvement_over_copy"] = 1-result["latent_mse"]/max(result["reference_mse"],1e-8)
    return result


@torch.inference_mode()
def sample_object_grid(*, unet, adapter, vae, sigma, latent_dataset, raw_dataset, output: Path, device, residual_scale: float, noise_scale: float, seed: int) -> None:
    chosen = list(range(min(8, len(raw_dataset))))
    reference = latent_dataset.payload["reference"][chosen].float().to(device)
    text = latent_dataset.payload["qwen_text"][chosen].float().to(device)
    condition = adapter.condition(text)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((len(chosen)//2, *reference.shape[1:]), generator=generator, device=device).repeat_interleave(2,0)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        latent = one_step_edit(unet, noise, reference, condition, sigma, residual_scale=residual_scale, noise_scale=noise_scale)
        generated = vae.decode(latent.to(vae.dtype)/vae.config.scaling_factor, return_dict=False)[0]
    generated = generated.float().clamp(-1,1).cpu()
    cell, labels = raw_dataset.image_size, 300
    canvas = Image.new("RGB", (labels+3*cell, len(chosen)*cell), "white"); draw = ImageDraw.Draw(canvas)
    for row,index in enumerate(chosen):
        raw=raw_dataset[index]
        # The entire decoded frame is evaluated and visualized.  Background
        # preservation is a learned constraint, never a hard pixel composite.
        for column,value in enumerate((raw["reference"],raw["target"],generated[row])):
            array=value.add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(array),(labels+column*cell,row*cell))
        draw.text((4,row*cell+4),raw["target_category"],fill="black")
        draw.text((4,row*cell+28),raw["prompt"][:46],fill="black")
        draw.text((4,row*cell+52),"input | target | generated",fill="black")
    output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output,quality=94,subsampling=0)


def train_object_model(
    *, initial_unet: Path, turbo_checkpoint: Path, latent_cache_dir: Path,
    data_dir: Path, instruction_cache: Path, adapter_checkpoint: Path,
    output_dir: Path, model_config: RepairModelConfig, training_config: SceneTrainingConfig,
    device: torch.device, warm_start_checkpoint: Path | None = None, seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True); _seed_everything(seed)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    datasets={name:ObjectLatentDataset(latent_cache_dir/f"{name}.pt") for name in ("train","val","test")}
    loaders={name:DataLoader(value,batch_size=2,shuffle=False,num_workers=training_config.num_workers,pin_memory=device.type=="cuda") for name,value in datasets.items()}
    adapter,_=_load_adapter(adapter_checkpoint,device); adapter.requires_grad_(True)
    router=TextObjectRouter().to(device)
    unet=UNet2DConditionModel.from_pretrained(initial_unet,subfolder="unet",variant="fp16",torch_dtype=torch.float32,local_files_only=True).to(device)
    expand_reference_conditioning(unet); optical,pruning=prepare_compact_optical_unet(unet,model_config)
    if warm_start_checkpoint:
        warm=torch.load(warm_start_checkpoint,map_location="cpu",weights_only=False,mmap=True)
        unet.load_state_dict(warm["unet"]); adapter.load_state_dict(warm["adapter"]); del warm
    unet.enable_gradient_checkpointing(); unet.requires_grad_(False)
    unet.conv_in.requires_grad_(True); unet.up_blocks.requires_grad_(True); unet.conv_norm_out.requires_grad_(True); unet.conv_out.requires_grad_(True)
    for _,p in optical.optical_parameters(): p.requires_grad_(True)
    optical_ids={id(p) for _,p in optical.optical_parameters()}
    optical_parameters=[p for p in unet.parameters() if p.requires_grad and id(p) in optical_ids]
    electronic_parameters=[p for p in unet.parameters() if p.requires_grad and id(p) not in optical_ids]
    optimizer=torch.optim.AdamW([
        {"params":electronic_parameters,"lr":training_config.learning_rate},
        {"params":optical_parameters,"lr":training_config.optical_learning_rate},
        {"params":adapter.parameters(),"lr":training_config.adapter_learning_rate},
        {"params":router.parameters(),"lr":training_config.adapter_learning_rate},
    ],weight_decay=training_config.weight_decay)
    scheduler=EulerDiscreteScheduler.from_pretrained(turbo_checkpoint,subfolder="scheduler",local_files_only=True);scheduler.set_timesteps(1,device=device);sigma=scheduler.sigmas[0]
    dtype=torch.float16 if device.type=="cuda" else torch.float32
    vae=AutoencoderKL.from_pretrained(turbo_checkpoint,subfolder="vae",variant="fp16",torch_dtype=dtype,local_files_only=True).to(device).eval().requires_grad_(False)
    raw_val=ProductObjectReplacementDataset(data_dir,"val",training_config.image_size,instruction_cache)
    initial=evaluate_object_model(unet,adapter,router,loaders["val"],sigma,device,residual_scale=training_config.residual_scale,noise_scale=training_config.noise_scale)
    best=math.inf;best_epoch=0;history=[];scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");started=time.perf_counter()
    for epoch in range(1,training_config.epochs+1):
        unet.train();adapter.train();router.train();optimizer.zero_grad(set_to_none=True);total=count=0
        for step,batch in enumerate(loaders["train"],1):
            reference=batch["reference"].to(device);target=batch["target"].to(device);edit=batch["edit_mask"].to(device);preserve=batch["preserve_mask"].to(device);text=batch["qwen_text"].to(device)
            labels=torch.as_tensor(batch["target_catalogue_indices"],device=device);condition=adapter.condition(text);noise=_paired_noise(reference)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                output=one_step_edit(unet,noise,reference,condition,sigma,residual_scale=training_config.residual_scale,noise_scale=training_config.noise_scale).float()
                loss=F.mse_loss(output,target)+training_config.background_weight*_masked_l1(output,target,edit)+training_config.foreground_preservation_weight*_masked_l1(output,reference,preserve)
                loss=loss+training_config.scene_router_weight*F.cross_entropy(router(text),labels)+training_config.detail_weight*latent_gradient_loss(output,target)
                decoded=vae.decode(output.to(vae.dtype)/vae.config.scaling_factor,return_dict=False)[0].float()
                with torch.no_grad(): decoded_target=vae.decode(target.to(vae.dtype)/vae.config.scaling_factor,return_dict=False)[0].float();decoded_ref=vae.decode(reference.to(vae.dtype)/vae.config.scaling_factor,return_dict=False)[0].float()
                pixel_edit=F.interpolate(edit.float(),decoded.shape[-2:],mode="bilinear",align_corners=False).clamp(0,1);pixel_preserve=1-pixel_edit
                loss=loss+training_config.pixel_background_weight*_masked_l1(decoded,decoded_target,pixel_edit)+training_config.pixel_foreground_weight*_masked_l1(decoded,decoded_ref,pixel_preserve)
                scaled=loss/training_config.gradient_accumulation
            scaler.scale(scaled).backward()
            if step%training_config.gradient_accumulation==0 or step==len(loaders["train"]):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([*electronic_parameters,*optical_parameters,*adapter.parameters(),*router.parameters()],1.0);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(reference);count+=len(reference)
        validation=evaluate_object_model(unet,adapter,router,loaders["val"],sigma,device,residual_scale=training_config.residual_scale,noise_scale=training_config.noise_scale)
        row={"epoch":epoch,"train_loss":total/count,"validation":validation,"optical_alpha":float(optical.fusion.alpha.detach())};history.append(row);print(json.dumps(row),flush=True)
        if validation["latent_mse"]<best:
            best=validation["latent_mse"];best_epoch=epoch
            torch.save({"schema_version":1,"epoch":epoch,"model_config":asdict(model_config),"training_config":asdict(training_config),"unet":{k:v.detach().half().cpu() for k,v in unet.state_dict().items()},"adapter":{k:v.detach().half().cpu() for k,v in adapter.state_dict().items()},"object_router":{k:v.detach().half().cpu() for k,v in router.state_dict().items()},"validation":validation},output_dir/"best_model.pt")
        sample_object_grid(unet=unet,adapter=adapter,vae=vae,sigma=sigma,latent_dataset=datasets["val"],raw_dataset=raw_val,output=output_dir/"samples"/f"epoch_{epoch:03d}.jpg",device=device,residual_scale=training_config.residual_scale,noise_scale=training_config.noise_scale,seed=seed+epoch)
    best_payload=torch.load(output_dir/"best_model.pt",map_location="cpu",weights_only=False,mmap=True);unet.load_state_dict(best_payload["unet"]);adapter.load_state_dict(best_payload["adapter"]);router.load_state_dict(best_payload["object_router"]);del best_payload
    test=evaluate_object_model(unet,adapter,router,loaders["test"],sigma,device,residual_scale=training_config.residual_scale,noise_scale=training_config.noise_scale)
    qwen=torch.load(instruction_cache,map_location="cpu",weights_only=False)["meta"]["qwen_pruning"]
    arch=architecture_report(unet,vae,adapter,router,optical,model_config);arch["vae_encoder_parameters"]=sum(p.numel() for p in vae.encoder.parameters())+sum(p.numel() for p in vae.quant_conv.parameters());arch["counted_qwen_parameters"]=qwen["counted_text_encoder_parameters"];arch["counted_end_to_end_parameters"]=arch["vae_encoder_parameters"]+arch["counted_qwen_parameters"]+arch["generation_tail_parameters"]
    report={"schema_version":1,"task":"text-guided object replacement","best_epoch":best_epoch,"initial_validation":initial,"test":test,"architecture":arch,"attention_pruning":pruning,"history":history,"training_seconds":time.perf_counter()-started,"gan_used":False,"inference_iterations":1}
    (output_dir/"training_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    del unet,vae,adapter,router,optimizer
    if device.type=="cuda":torch.cuda.empty_cache()
    return report


__all__=["ObjectLatentDataset","TextObjectRouter","cache_object_latents","evaluate_object_model","load_scene_config","sample_object_grid","train_object_model"]
