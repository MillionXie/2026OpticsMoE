"""Cached-teacher progressive distillation for a sub-300M full-frame editor."""

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
from .product_repair_model import RepairModelConfig, one_step_edit
from .product_scene_replace_data import ProductBackgroundReplacementDataset
from .product_scene_replace_training import ReplacementLatentDataset, TextAttributeRouter, _attribute_loss, _labels
from .product_scene_training import SceneTrainingConfig, _paired_noise, _seed_everything, load_scene_config
from .progressive_student import (
    PROGRESSIVE_WIDTHS,
    build_narrow_optical_unet,
    copy_overlapping_state,
    counted_student_parameters,
)


class DistillationDataset(Dataset[dict[str, Any]]):
    def __init__(self, student_path: Path, teacher_path: Path) -> None:
        self.student = torch.load(student_path, map_location="cpu", weights_only=False, mmap=True)
        self.teacher = torch.load(teacher_path, map_location="cpu", weights_only=False, mmap=True)
        if len(self.student["reference"]) != len(self.teacher["prediction"]):
            raise ValueError("Student examples and teacher predictions have different lengths")

    def __len__(self) -> int:
        return len(self.student["reference"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = {
            key: self.student[key][index].float()
            for key in ("reference", "target", "foreground_mask", "background_mask", "qwen_text")
        }
        value["teacher_prediction"] = self.teacher["prediction"][index].float()
        value["noise"] = self.teacher["noise"][index].float()
        for key in ("rooms", "tones", "brightness", "directions"):
            value[key] = self.student[key][index]
        return value


@torch.inference_mode()
def build_teacher_cache(
    *, teacher_checkpoint: Path, adapter_checkpoint: Path, initial_unet: Path,
    teacher_latent_dir: Path, output_dir: Path, turbo_checkpoint: Path,
    device: torch.device, batch_size: int = 4, num_workers: int = 4,
) -> dict[str, Any]:
    """Cache final teacher latents and the exact Gaussian input used for them."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from diffusers import EulerDiscreteScheduler, UNet2DConditionModel
    from .product_repair_model import expand_reference_conditioning

    payload = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_config = RepairModelConfig(**payload["model_config"])
    training = payload["training_config"]
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"]); adapter.eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    expand_reference_conditioning(unet); prepare_compact_optical_unet(unet, model_config)
    unet.load_state_dict(payload["unet"]); unet.eval().requires_grad_(False)
    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True,
    )
    scheduler.set_timesteps(1, device=device); sigma = scheduler.sigmas[0]
    summary: dict[str, Any] = {"schema_version": 1, "teacher": str(teacher_checkpoint), "splits": {}}
    for split_index, split in enumerate(("train", "val", "test")):
        dataset = ReplacementLatentDataset(teacher_latent_dir / f"{split}.pt")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        predictions, noises = [], []
        generator = torch.Generator(device=device).manual_seed(7300 + split_index)
        for batch in loader:
            reference = batch["reference"].to(device)
            text = batch["qwen_text"].to(device)
            noise = _paired_noise(reference, generator)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = one_step_edit(
                    unet, noise, reference, adapter.condition(text), sigma,
                    residual_scale=float(training["residual_scale"]),
                    noise_scale=float(training["noise_scale"]),
                )
            predictions.append(prediction.half().cpu()); noises.append(noise.half().cpu())
        packed = {"prediction": torch.cat(predictions), "noise": torch.cat(noises)}
        torch.save(packed, output_dir / f"{split}.pt")
        summary["splits"][split] = len(dataset)
    (output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    del unet, adapter, payload
    if device.type == "cuda": torch.cuda.empty_cache()
    return summary


@torch.inference_mode()
def _evaluate(unet, adapter, router, loader, sigma, device, training: SceneTrainingConfig) -> dict[str, float]:
    unet.eval(); adapter.eval(); router.eval()
    totals = {"latent_mse": 0.0, "teacher_mse": 0.0, "latent_l1": 0.0, "exact_combination_accuracy": 0.0}
    count = 0
    for batch in loader:
        reference=batch["reference"].to(device);target=batch["target"].to(device)
        teacher=batch["teacher_prediction"].to(device);noise=batch["noise"].to(device);text=batch["qwen_text"].to(device)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=="cuda"):
            output=one_step_edit(unet,noise,reference,adapter.condition(text),sigma,residual_scale=training.residual_scale,noise_scale=training.noise_scale).float()
        size=len(reference);totals["latent_mse"]+=float(F.mse_loss(output,target))*size
        totals["teacher_mse"]+=float(F.mse_loss(output,teacher))*size;totals["latent_l1"]+=float(F.l1_loss(output,target))*size
        logits=router(text);labels=_labels(batch,device)
        totals["exact_combination_accuracy"]+=float(torch.stack([(logits[k].argmax(-1)==labels[k]) for k in logits]).all(0).float().mean())*size
        count+=size
    return {key:value/count for key,value in totals.items()}


@torch.inference_mode()
def _sample_full_frame(*,unet,adapter,vae,sigma,dataset,raw_dataset,output:Path,device,training:SceneTrainingConfig)->None:
    chosen=list(range(raw_dataset.targets_per_source));reference=dataset.student["reference"][chosen].float().to(device)
    text=dataset.student["qwen_text"][chosen].float().to(device);noise=dataset.teacher["noise"][chosen].float().to(device)
    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=="cuda"):
        latent=one_step_edit(unet,noise,reference,adapter.condition(text),sigma,residual_scale=training.residual_scale,noise_scale=training.noise_scale)
        generated=vae.decode(latent.to(vae.dtype)/vae.config.scaling_factor,return_dict=False)[0]
    generated=generated.float().clamp(-1,1).cpu();cell=raw_dataset.image_size;label=300
    canvas=Image.new("RGB",(label+3*cell,len(chosen)*cell),"white");draw=ImageDraw.Draw(canvas)
    for row,index in enumerate(chosen):
        raw=raw_dataset[index]
        for column,value in enumerate((raw["reference"],raw["target"],generated[row])):
            array=value.add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(array),(label+column*cell,row*cell))
        draw.text((4,row*cell+4),raw["target_scene_id"].replace("__"," / "),fill="black")
        draw.text((4,row*cell+28),raw["prompt"][:46],fill="black")
        draw.text((4,row*cell+52),"input | target | full generated",fill="black")
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output,quality=94,subsampling=0)


def train_progressive_student(
    *, initial_unet: Path, turbo_checkpoint: Path, student_latent_dir: Path,
    teacher_cache_dir: Path, data_dir: Path, instruction_cache: Path,
    adapter_checkpoint: Path, teacher_checkpoint: Path, output_dir: Path,
    model_config: RepairModelConfig, training_config: SceneTrainingConfig,
    device: torch.device, stage_epochs: tuple[int, ...] = (1,1,2), seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(output_dir)
    if len(stage_epochs)!=len(PROGRESSIVE_WIDTHS): raise ValueError("stage_epochs must match progressive widths")
    output_dir.mkdir(parents=True);_seed_everything(seed)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    base_config=UNet2DConditionModel.load_config(initial_unet,subfolder="unet",local_files_only=True)
    teacher=torch.load(teacher_checkpoint,map_location="cpu",weights_only=False,mmap=True)
    datasets={name:DistillationDataset(student_latent_dir/f"{name}.pt",teacher_cache_dir/f"{name}.pt") for name in ("train","val","test")}
    loaders={name:DataLoader(value,batch_size=training_config.batch_size,shuffle=name=="train",num_workers=training_config.num_workers,pin_memory=device.type=="cuda") for name,value in datasets.items()}
    scheduler=EulerDiscreteScheduler.from_pretrained(turbo_checkpoint,subfolder="scheduler",local_files_only=True);scheduler.set_timesteps(1,device=device);sigma=scheduler.sigmas[0]
    dtype=torch.float16 if device.type=="cuda" else torch.float32
    vae=AutoencoderKL.from_pretrained(turbo_checkpoint,subfolder="vae",variant="fp16",torch_dtype=dtype,local_files_only=True).to(device).eval().requires_grad_(False)
    raw_val=ProductBackgroundReplacementDataset(data_dir,"val",training_config.image_size,instruction_cache)
    qwen_meta=torch.load(instruction_cache,map_location="cpu",weights_only=False)["meta"]["qwen_pruning"]
    source_unet=teacher["unet"];source_adapter=teacher["adapter"];source_router=teacher["attribute_router"]
    stages=[];started=time.perf_counter();final_payload=None
    for stage_index,(widths,epochs) in enumerate(zip(PROGRESSIVE_WIDTHS,stage_epochs),1):
        unet,optical,pruning=build_narrow_optical_unet(base_config,widths,model_config);warm=copy_overlapping_state(unet,source_unet);unet=unet.to(device)
        adapter,_=_load_adapter(adapter_checkpoint,device);adapter.load_state_dict(source_adapter);adapter.requires_grad_(True)
        router=TextAttributeRouter(datasets["train"].student["qwen_text"].shape[1]).to(device);router.load_state_dict(source_router);router.requires_grad_(True)
        unet.enable_gradient_checkpointing();unet.requires_grad_(True)
        optical_ids={id(p) for p in optical.parameters()};electronic=[p for p in unet.parameters() if id(p) not in optical_ids];optical_params=list(optical.parameters())
        optimizer=torch.optim.AdamW([
            {"params":electronic,"lr":training_config.learning_rate},
            {"params":optical_params,"lr":training_config.optical_learning_rate},
            {"params":adapter.parameters(),"lr":training_config.adapter_learning_rate},
            {"params":router.parameters(),"lr":training_config.adapter_learning_rate},
        ],weight_decay=training_config.weight_decay)
        scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");best=math.inf;history=[];stage_dir=output_dir/f"stage_{stage_index}_{widths[-1]}";stage_dir.mkdir()
        for epoch in range(1,epochs+1):
            unet.train();adapter.train();router.train();optimizer.zero_grad(set_to_none=True);running=samples=0
            for step,batch in enumerate(loaders["train"],1):
                reference=batch["reference"].to(device);target=batch["target"].to(device);teacher_prediction=batch["teacher_prediction"].to(device);noise=batch["noise"].to(device);text=batch["qwen_text"].to(device)
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                    output=one_step_edit(unet,noise,reference,adapter.condition(text),sigma,residual_scale=training_config.residual_scale,noise_scale=training_config.noise_scale).float()
                    logits=router(text);labels=_labels(batch,device)
                    loss=F.mse_loss(output,target)+0.75*F.mse_loss(output,teacher_prediction)+0.35*F.l1_loss(output,target)
                    loss=loss+training_config.detail_weight*latent_gradient_loss(output,target)+training_config.scene_router_weight*_attribute_loss(logits,labels)
                    scaled=loss/training_config.gradient_accumulation
                scaler.scale(scaled).backward()
                if step%training_config.gradient_accumulation==0 or step==len(loaders["train"]):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([*electronic,*optical_params,*adapter.parameters(),*router.parameters()],1.0);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                running+=float(loss.detach())*len(reference);samples+=len(reference)
            validation=_evaluate(unet,adapter,router,loaders["val"],sigma,device,training_config);row={"epoch":epoch,"loss":running/samples,"validation":validation,"alpha":float(optical.fusion.alpha.detach())};history.append(row);print(json.dumps({"stage":stage_index,**row}),flush=True)
            if validation["latent_mse"]<best:
                best=validation["latent_mse"]
                final_payload={"schema_version":1,"student_widths":list(widths),"model_config":asdict(model_config),"training_config":asdict(training_config),"unet":{k:v.detach().half().cpu() for k,v in unet.state_dict().items()},"adapter":{k:v.detach().half().cpu() for k,v in adapter.state_dict().items()},"attribute_router":{k:v.detach().half().cpu() for k,v in router.state_dict().items()},"validation":validation,"qwen_layers":2}
                torch.save(final_payload,stage_dir/"best_model.pt")
        source_unet=final_payload["unet"];source_adapter=final_payload["adapter"];source_router=final_payload["attribute_router"]
        counts=counted_student_parameters(unet=unet,adapter=adapter,router=router,qwen_counted=qwen_meta["counted_text_encoder_parameters"],vae_encoder=sum(p.numel() for p in vae.encoder.parameters())+sum(p.numel() for p in vae.quant_conv.parameters()),vae_decoder=sum(p.numel() for p in vae.decoder.parameters())+sum(p.numel() for p in vae.post_quant_conv.parameters()))
        stages.append({"stage":stage_index,"widths":list(widths),"epochs":epochs,"warm_start":warm,"pruning":pruning,"best_validation_mse":best,"history":history,"parameters":counts})
        del optimizer,unet,adapter,router,optical;torch.cuda.empty_cache() if device.type=="cuda" else None
    final_widths=tuple(final_payload["student_widths"]);unet,optical,_=build_narrow_optical_unet(base_config,final_widths,model_config);unet.load_state_dict(final_payload["unet"]);unet=unet.to(device)
    adapter,_=_load_adapter(adapter_checkpoint,device);adapter.load_state_dict(final_payload["adapter"]);router=TextAttributeRouter(datasets["train"].student["qwen_text"].shape[1]).to(device);router.load_state_dict(final_payload["attribute_router"])
    test=_evaluate(unet,adapter,router,loaders["test"],sigma,device,training_config)
    _sample_full_frame(unet=unet,adapter=adapter,vae=vae,sigma=sigma,dataset=datasets["val"],raw_dataset=raw_val,output=output_dir/"full_frame_grid.jpg",device=device,training=training_config)
    torch.save(final_payload,output_dir/"best_model.pt")
    report={"schema_version":1,"task":"full-frame background replacement via progressive local distillation","teacher":str(teacher_checkpoint),"stages":stages,"test":test,"parameters":stages[-1]["parameters"],"hard_pixel_composite":False,"gan_used":False,"inference_iterations":1,"training_seconds":time.perf_counter()-started}
    (output_dir/"training_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    del unet,adapter,router,vae,teacher
    if device.type=="cuda":torch.cuda.empty_cache()
    return report


__all__=["DistillationDataset","build_teacher_cache","train_progressive_student","load_scene_config"]
