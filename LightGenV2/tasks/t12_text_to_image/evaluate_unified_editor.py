"""Image-space evaluation of one unified Qwen-mini product editor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .qwen_mini_small import PromptEmbeddingLookup, QwenMiniConfig, QwenMiniTextEncoder
from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor, build_dataset, _edge


class TorchvisionInceptionFeatures(nn.Module):
    """2048-D ImageNet Inception-v3 features for an explicitly labeled FID variant."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.backbone = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.backbone.transform_input = False
        self.backbone.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std", torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_device = next(self.backbone.parameters()).device
        value = F.interpolate(images.to(feature_device).float().div(255), size=(299,299), mode="bilinear", align_corners=False)
        return self.backbone((value-self.mean.to(feature_device))/self.std.to(feature_device)).to(images.device)


@torch.inference_mode()
def evaluate_small(*, checkpoint: Path, data_dir: Path, instruction_cache: Path,
                   embedding_cache: Path, output: Path, device: torch.device,
                   batch_size: int = 16, seed: int = 42,
                   distribution_metrics: bool = False) -> dict:
    if distribution_metrics:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    editor_values = dict(payload["editor_config"])
    editor_values["widths"] = tuple(editor_values["widths"])
    editor_config = SmallEditorConfig(**editor_values)
    model = SmallFullFrameEditor(editor_config)
    model.text = QwenMiniTextEncoder(QwenMiniConfig(**payload["qwen_mini_config"]))
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    dataset = build_dataset("unified", data_dir, "test", editor_config.image_size, instruction_cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    lookup = PromptEmbeddingLookup(embedding_cache)
    extractor = TorchvisionInceptionFeatures().to(device) if distribution_metrics else None
    fid = FrechetInceptionDistance(feature=extractor).to(device) if distribution_metrics else None
    kid = KernelInceptionDistance(feature=extractor, subset_size=100, subsets=10).to(device) if distribution_metrics else None
    metrics = {mode: {"n": 0, "mse": 0., "l1": 0., "edge_l1": 0.,
                      "copy_mse": 0.}
               for mode in ("background", "object", "joint")}
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        reference = batch["reference"].to(device)
        target = batch["target"].to(device)
        embeddings, mask, _ = lookup.batch(list(batch["prompt"]), device)
        noise = torch.randn(reference.shape, generator=generator, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            generated = model(reference, (embeddings, mask), noise, None).float()
        if distribution_metrics:
            real_uint8 = target.add(1).mul(127.5).clamp(0, 255).byte()
            fake_uint8 = generated.add(1).mul(127.5).clamp(0, 255).byte()
            fid.update(real_uint8, real=True); fid.update(fake_uint8, real=False)
            kid.update(real_uint8, real=True); kid.update(fake_uint8, real=False)
        for mode in metrics:
            indices = [i for i, name in enumerate(batch["mode"]) if name == mode]
            if not indices:
                continue
            selected = torch.as_tensor(indices, device=device)
            source, wanted, output_image = reference[selected], target[selected], generated[selected]
            row = metrics[mode]; count = len(indices); row["n"] += count
            row["mse"] += float(F.mse_loss(output_image, wanted)) * count
            row["l1"] += float(F.l1_loss(output_image, wanted)) * count
            row["edge_l1"] += float(F.l1_loss(_edge(output_image), _edge(wanted))) * count
            row["copy_mse"] += float(F.mse_loss(source, wanted)) * count
    for row in metrics.values():
        for key in ("mse", "l1", "edge_l1", "copy_mse"):
            row[key] /= row["n"]
        row["improvement_over_copy"] = 1 - row["mse"] / row["copy_mse"]
    result = {"schema_version": 1, "checkpoint": str(checkpoint),
              "counted_parameters": int(payload["counted_parameters"]),
              "resolution": editor_config.image_size,
              "test_pairs": len(dataset), "source_views": len(dataset.base.sources),
              "fixed_target_designs": len(dataset.base.catalogue),
              "metrics_by_mode": metrics,
              "fid_caveat": "Repeated paired targets and four fixed object designs; not open-set FID"}
    if distribution_metrics:
        kid_mean, kid_std = kid.compute()
        result.update(fid=float(fid.compute()), kid_mean=float(kid_mean), kid_std=float(kid_std))
        result["feature_extractor"] = "torchvision Inception-v3 ImageNet1K; differs from canonical TensorFlow FID"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "data-dir", "instruction-cache", "embedding-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distribution-metrics", action="store_true")
    args = parser.parse_args()
    result = evaluate_small(checkpoint=args.checkpoint.resolve(), data_dir=args.data_dir.resolve(),
                            instruction_cache=args.instruction_cache.resolve(),
                            embedding_cache=args.embedding_cache.resolve(), output=args.output.resolve(),
                            device=torch.device(args.device), batch_size=args.batch_size, seed=args.seed,
                            distribution_metrics=args.distribution_metrics)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
