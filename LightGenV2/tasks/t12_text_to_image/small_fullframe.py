"""Sub-50M direct-RGB student for the three T12 conditional generation tasks."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .modeling import CompactFourierOptics, ConditionedElectronicResidual, ScaleMatchedFusion
from .premium_material_data import PremiumMaterialDataset
from .product_global_redesign_data import ProductGlobalRedesignDataset
from .product_scene_replace_data import ProductBackgroundReplacementDataset


@dataclass(frozen=True)
class SmallEditorConfig:
    image_size: int = 128
    widths: tuple[int, int, int, int] = (48, 96, 160, 224)
    text_width: int = 96
    condition_dim: int = 160
    max_text_bytes: int = 160
    optical_experts: int = 4
    optical_top_k: int = 2
    alpha_initial: float = .50
    alpha_minimum: float = .40
    alpha_maximum: float = .75
    residual_limit: float = 2.0
    control_classes: int = 0


class ByteTextEncoder(nn.Module):
    """Tokenizer-free prompt encoder; UTF-8 bytes are the input vocabulary."""

    def __init__(self, width: int, condition_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.gru = nn.GRU(width, condition_dim, batch_first=True)
        self.norm = nn.LayerNorm(condition_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        lengths = tokens.ne(0).sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.norm(hidden[-1])


def encode_prompts(prompts: list[str] | tuple[str, ...], max_length: int, device: torch.device) -> torch.Tensor:
    result = torch.zeros(len(prompts), max_length, dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        values = list(prompt.encode("utf-8")[:max_length])
        if values:
            result[row, :len(values)] = torch.as_tensor(values, device=device) + 1
    return result


def premium_control_ids_from_prompts(prompts: list[str] | tuple[str, ...], device: torch.device) -> torch.Tensor:
    """Parse the controlled premium vocabulary without a large language model."""
    categories = ("lamp", "table", "backpack")
    style_terms = (
        ("brass", "walnut", "cognac", "leather"),
        ("obsidian", "black", "nylon"),
        ("ivory", "ceramic", "travertine", "canvas"),
        ("teal", "glass", "lacquer", "textile"),
    )
    values = []
    for prompt in prompts:
        text = prompt.lower()
        category = next((index for index, name in enumerate(categories) if name in text), None)
        style = next((index for index, terms in enumerate(style_terms) if any(term in text for term in terms)), None)
        if category is None or style is None:
            raise ValueError(f"Prompt is outside the controlled premium vocabulary: {prompt!r}")
        values.append(category * 4 + style)
    return torch.as_tensor(values, dtype=torch.long, device=device)


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = math.gcd(output_channels, 16)
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1),
            nn.GroupNorm(groups, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.SiLU(),
        )
        self.skip = nn.Conv2d(input_channels, output_channels, 1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (self.main(value) + self.skip(value)) * (2**-.5)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = math.gcd(output_channels, 16)
        self.conv1 = nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, output_channels, affine=False)
        self.norm2 = nn.GroupNorm(groups, output_channels, affine=False)
        self.affine = nn.Linear(condition_dim, 4 * output_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = torch.cat((value, skip), dim=1)
        scale1, shift1, scale2, shift2 = self.affine(condition).chunk(4, 1)
        value = self.norm1(self.conv1(value)) * (1+scale1[:, :, None, None]) + shift1[:, :, None, None]
        value = F.silu(value)
        value = self.norm2(self.conv2(value)) * (1+scale2[:, :, None, None]) + shift2[:, :, None, None]
        return F.silu(value)


class ParallelSmallBottleneck(nn.Module):
    """Electronic and optical branches consume the same encoder feature map."""

    def __init__(self, width: int, condition_dim: int, grid: int, config: SmallEditorConfig) -> None:
        super().__init__()
        self.grid = grid
        self.electronic = ConditionedElectronicResidual(width, condition_dim, grid)
        self.optical_condition = nn.Linear(condition_dim, 2*width)
        self.optical = CompactFourierOptics(width, grid, config.optical_experts, config.optical_top_k)
        self.expert_gate = nn.Parameter(torch.tensor(-1.5)); self.global_gate = nn.Parameter(torch.tensor(-1.5))
        self.fusion = ScaleMatchedFusion(config.alpha_initial, config.alpha_minimum, config.alpha_maximum, 1e-6)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        tokens = value.flatten(2).transpose(1, 2)
        electronic = self.electronic(tokens, condition)
        scale, shift = self.optical_condition(condition).chunk(2, 1)
        optical_input = tokens * (1+scale[:, None]) + shift[:, None]
        expert = self.optical.expert(optical_input)
        optical = tokens + torch.sigmoid(self.expert_gate) * expert
        optical = optical + torch.sigmoid(self.global_gate) * self.optical.global_block(optical)
        return self.fusion(electronic, optical).transpose(1, 2).reshape_as(value)


class SmallFullFrameEditor(nn.Module):
    def __init__(self, config: SmallEditorConfig) -> None:
        super().__init__(); self.config = config
        widths = config.widths
        self.text = ByteTextEncoder(config.text_width, config.condition_dim)
        self.control_embedding = nn.Embedding(config.control_classes, config.condition_dim) if config.control_classes else None
        self.stem = nn.Sequential(nn.Conv2d(6, widths[0], 3, padding=1), nn.SiLU())
        self.down1 = DownBlock(widths[0], widths[1]); self.down2 = DownBlock(widths[1], widths[2]); self.down3 = DownBlock(widths[2], widths[3])
        self.bottleneck = ParallelSmallBottleneck(widths[3], config.condition_dim, config.image_size//8, config)
        self.up3 = UpBlock(widths[3], widths[2], widths[2], config.condition_dim)
        self.up2 = UpBlock(widths[2], widths[1], widths[1], config.condition_dim)
        self.up1 = UpBlock(widths[1], widths[0], widths[0], config.condition_dim)
        self.to_delta = nn.Sequential(nn.Conv2d(widths[0], widths[0], 3, padding=1), nn.SiLU(), nn.Conv2d(widths[0], 3, 3, padding=1))
        nn.init.zeros_(self.to_delta[-1].weight); nn.init.zeros_(self.to_delta[-1].bias)

    def encode_condition(self, prompt_tokens: torch.Tensor, control_ids: torch.Tensor | None = None) -> torch.Tensor:
        condition = self.text(prompt_tokens)
        if self.control_embedding is not None:
            if control_ids is None:
                raise ValueError("This controlled student requires parsed prompt control IDs")
            condition = condition + self.control_embedding(control_ids)
        return condition

    def forward(self, reference: torch.Tensor, prompt_tokens: torch.Tensor, noise: torch.Tensor, control_ids: torch.Tensor | None = None) -> torch.Tensor:
        condition = self.encode_condition(prompt_tokens, control_ids)
        s0 = self.stem(torch.cat((reference, .08*noise), dim=1)); s1 = self.down1(s0); s2 = self.down2(s1)
        value = self.bottleneck(self.down3(s2), condition)
        value = self.up3(value, s2, condition); value = self.up2(value, s1, condition); value = self.up1(value, s0, condition)
        delta = torch.tanh(self.to_delta(value)) * self.config.residual_limit
        # Identity-biased parameterisation, not pixel compositing: every pixel
        # is predicted by the network and can move across the full output range.
        base = torch.atanh(reference.float().clamp(-.98, .98))
        return torch.tanh(base + delta.float())


class PromptPairDataset(Dataset[dict[str, Any]]):
    def __init__(self, base: Dataset[dict[str, Any]]) -> None:
        self.base = base
    def __len__(self) -> int: return len(self.base)
    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self.base[index]
        if "catalogue_index" in value:
            condition_index = int(value["catalogue_index"])
        else:
            condition_index = (
                int(value["room_index"]) * 12 + int(value["tone_index"]) * 4
                + int(value["brightness_index"]) * 2 + int(value["direction_index"])
            )
        return {
            "reference": value["reference"], "target": value["target"],
            "prompt": value["prompt"], "sample_id": value["sample_id"],
            "condition_index": condition_index,
        }


DATASETS = {
    "background": ProductBackgroundReplacementDataset,
    "redesign": ProductGlobalRedesignDataset,
    "premium": PremiumMaterialDataset,
}


def build_dataset(task: str, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> PromptPairDataset:
    return PromptPairDataset(DATASETS[task](data_dir, split, image_size, instruction_cache))


def _edge(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(1, keepdim=True)
    kernel = value.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).reshape(1,1,3,3)
    return torch.sqrt(F.conv2d(gray, kernel, padding=1).square()+F.conv2d(gray, kernel.transpose(-1,-2), padding=1).square()+1e-6)


@torch.inference_mode()
def evaluate(model: SmallFullFrameEditor, loader: DataLoader, device: torch.device, seed: int) -> dict[str, float]:
    model.eval(); totals={"mse":0.0,"l1":0.0,"edge_l1":0.0,"copy_mse":0.0}; count=0
    generator=torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        reference=batch["reference"].to(device);target=batch["target"].to(device)
        tokens=encode_prompts(list(batch["prompt"]),model.config.max_text_bytes,device)
        noise=torch.randn(reference.shape,generator=generator,device=device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            controls=batch["condition_index"].to(device) if model.control_embedding is not None else None
            output=model(reference,tokens,noise,controls)
        size=len(reference);totals["mse"]+=float(F.mse_loss(output,target))*size;totals["l1"]+=float(F.l1_loss(output,target))*size
        totals["edge_l1"]+=float(F.l1_loss(_edge(output),_edge(target)))*size;totals["copy_mse"]+=float(F.mse_loss(reference,target))*size;count+=size
    result={key:value/count for key,value in totals.items()};result["mse_improvement_over_copy"]=1-result["mse"]/max(result["copy_mse"],1e-8)
    return result


@torch.inference_mode()
def save_samples(model: SmallFullFrameEditor, dataset: PromptPairDataset, output: Path, device: torch.device, seed: int) -> None:
    model.eval(); count=min(12,len(dataset)); step=max(1,len(dataset)//count); indices=[min(i*step,len(dataset)-1) for i in range(count)]
    items=[dataset[index] for index in indices]; reference=torch.stack([item["reference"] for item in items]).to(device)
    tokens=encode_prompts([item["prompt"] for item in items],model.config.max_text_bytes,device);generator=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn(reference.shape,generator=generator,device=device)
    controls=torch.as_tensor([item["condition_index"] for item in items],device=device) if model.control_embedding is not None else None
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): prediction=model(reference,tokens,noise,controls)
    cell=model.config.image_size; label=280; canvas=Image.new("RGB",(label+3*cell,count*cell),"white");draw=ImageDraw.Draw(canvas)
    for row,item in enumerate(items):
        for column,value in enumerate((item["reference"],item["target"],prediction[row].cpu())):
            array=value.add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy();canvas.paste(Image.fromarray(array),(label+column*cell,row*cell))
        draw.text((4,row*cell+4),item["prompt"][:44],fill="black");draw.text((4,row*cell+28),"input | target | student",fill="black")
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output,quality=94,subsampling=0)


def train_small_editor(
    *, task: str, data_dir: Path, instruction_cache: Path, output_dir: Path,
    device: torch.device, epochs: int = 20, batch_size: int = 16,
    learning_rate: float = 2e-4, seed: int = 42,
    structured_control: bool = False,
) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    config=SmallEditorConfig(control_classes=64 if structured_control else 0);datasets={name:build_dataset(task,data_dir,name,config.image_size,instruction_cache) for name in ("train","val","test")}
    loaders={name:DataLoader(value,batch_size=batch_size,shuffle=name=="train",num_workers=4,pin_memory=True,persistent_workers=True) for name,value in datasets.items()}
    model=SmallFullFrameEditor(config).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>=50_000_000: raise AssertionError(parameters)
    # Auxiliary prompt classification is used only during training.  It forces
    # the tiny byte encoder to separate material/scene phrases instead of
    # averaging visually different targets; it adds no inference parameters.
    text_classifier=nn.Linear(config.condition_dim,64).to(device)
    optimizer=torch.optim.AdamW([*model.parameters(),*text_classifier.parameters()],lr=learning_rate,weight_decay=1e-2);scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    initial=evaluate(model,loaders["val"],device,seed+100);best=math.inf;best_epoch=0;history=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();total=samples=0
        for batch in loaders["train"]:
            reference=batch["reference"].to(device);target=batch["target"].to(device);tokens=encode_prompts(list(batch["prompt"]),config.max_text_bytes,device);noise=torch.randn_like(reference);labels=batch["condition_index"].to(device)
            if random.random()<.5:reference=reference.flip(-1);target=target.flip(-1);noise=noise.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                controls=labels if model.control_embedding is not None else None
                output=model(reference,tokens,noise,controls)
                text_logits=text_classifier(model.text(tokens))
                loss=F.l1_loss(output,target)+.65*F.mse_loss(output,target)+.10*F.l1_loss(_edge(output),_edge(target))+.20*F.cross_entropy(text_logits.float(),labels)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([*model.parameters(),*text_classifier.parameters()],3.0);scaler.step(optimizer);scaler.update()
            with torch.no_grad():
                for target_parameter,source_parameter in zip(ema.parameters(),model.parameters()):target_parameter.lerp_(source_parameter,.01)
                for target_buffer,source_buffer in zip(ema.buffers(),model.buffers()):target_buffer.copy_(source_buffer)
            total+=float(loss.detach())*len(reference);samples+=len(reference)
        validation=evaluate(ema,loaders["val"],device,seed+epoch);row={"epoch":epoch,"train_loss":total/samples,"validation":validation,"alpha":float(ema.bottleneck.fusion.alpha)};history.append(row);print(json.dumps(row),flush=True)
        if validation["mse"]<best:
            best=validation["mse"];best_epoch=epoch
            prompt_encoding="UTF-8 bytes + deterministic category/material control parser" if structured_control else "UTF-8 bytes; no Qwen/token embedding"
            torch.save({"schema_version":1,"task":task,"config":{**asdict(config),"widths":list(config.widths)},"model":{key:value.detach().half().cpu() for key,value in ema.state_dict().items()},"parameters":parameters,"epoch":epoch,"validation":validation,"prompt_encoding":prompt_encoding,"hard_pixel_composite":False,"gan_used":False,"inference_iterations":1},output_dir/"best_model.pt")
        if epoch in {1,5,10,epochs}:save_samples(ema,datasets["val"],output_dir/"samples"/f"epoch_{epoch:03d}.jpg",device,seed+epoch)
    payload=torch.load(output_dir/"best_model.pt",map_location="cpu",weights_only=False);ema.load_state_dict(payload["model"]);test=evaluate(ema,loaders["test"],device,seed+999)
    report={"schema_version":1,"task":task,"best_epoch":best_epoch,"parameters":parameters,"under_50m":parameters<50_000_000,"initial_validation":initial,"test":test,"history":history,"training_seconds":time.perf_counter()-started,"optical_alpha":float(ema.bottleneck.fusion.alpha),"hard_pixel_composite":False,"gan_used":False,"inference_iterations":1,"resolution":config.image_size}
    (output_dir/"training_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");save_samples(ema,datasets["test"],output_dir/"final_grid.jpg",device,seed+999)
    del model,ema,optimizer,text_classifier
    if device.type=="cuda":torch.cuda.empty_cache()
    return report


__all__=["SmallEditorConfig","SmallFullFrameEditor","encode_prompts","premium_control_ids_from_prompts","train_small_editor"]
