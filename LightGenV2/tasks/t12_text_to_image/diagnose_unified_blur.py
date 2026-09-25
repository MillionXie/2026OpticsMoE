"""Measure the frozen VAE's image detail ceiling on unified edit targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .product_unified_edit_data import UnifiedProductEditDataset
from .small_fullframe import _edge


@torch.inference_mode()
def diagnose(*, data_dir: Path, instruction_cache: Path, vae_checkpoint: Path,
             output_dir: Path, device: torch.device, samples_per_mode: int = 4) -> dict:
    from diffusers import AutoencoderKL

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_checkpoint, subfolder="vae", variant="fp16",
                                       torch_dtype=dtype, local_files_only=True).to(device).eval()
    data = UnifiedProductEditDataset(data_dir, "test", 256, instruction_cache)
    selected = {mode: [] for mode in ("background", "object", "joint")}
    for index in range(len(data)):
        mode = data[index]["mode"]
        if len(selected[mode]) < samples_per_mode:
            selected[mode].append(index)
        if all(len(items) == samples_per_mode for items in selected.values()):
            break
    rows = [(mode, i) for mode, indices in selected.items() for i in indices]
    canvas = Image.new("RGB", (220+2*256, len(rows)*256), "white")
    draw = ImageDraw.Draw(canvas)
    totals = {mode: {"n": 0, "mse": 0., "l1": 0., "edge_l1": 0.}
              for mode in selected}
    for row, (mode, index) in enumerate(rows):
        item = data[index]
        target = item["target"].unsqueeze(0).to(device)
        latent = vae.encode(target.to(dtype)).latent_dist.mode()
        restored = vae.decode(latent).sample.float().clamp(-1, 1)
        record = totals[mode]; record["n"] += 1
        record["mse"] += float(F.mse_loss(restored, target))
        record["l1"] += float(F.l1_loss(restored, target))
        record["edge_l1"] += float(F.l1_loss(_edge(restored), _edge(target)))
        for col, image in enumerate((target[0], restored[0])):
            rgb = image.add(1).mul(127.5).clamp(0, 255).byte().permute(1,2,0).cpu().numpy()
            canvas.paste(Image.fromarray(rgb), (220+256*col, 256*row))
        draw.text((6, row*256+6), mode, fill="black")
        draw.text((6, row*256+27), "target | VAE reconstruction", fill="black")
    for record in totals.values():
        for key in ("mse", "l1", "edge_l1"):
            record[key] /= record["n"]
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir/"vae_reconstruction_grid.jpg", quality=95)
    report = {"schema_version": 1, "resolution": 256, "sampled_per_mode": samples_per_mode,
              "metrics_by_mode": totals, "grid": str(output_dir/"vae_reconstruction_grid.jpg")}
    (output_dir/"vae_reconstruction_metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("data-dir","instruction-cache","vae-checkpoint","output-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    report=diagnose(data_dir=args.data_dir.resolve(), instruction_cache=args.instruction_cache.resolve(),
                    vae_checkpoint=args.vae_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
                    device=torch.device(args.device))
    print(json.dumps(report,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
