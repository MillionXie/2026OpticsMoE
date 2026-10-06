"""Paired TEST metric audit for the frozen T12 five-model comparison.

Uses saved RGB PNGs only. Evaluation masks are reconstructed from dataset manifests;
they are never sent to any model. The experiment is read-only except --output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape != (256, 256, 3):
        raise ValueError(f"unexpected image shape {path}: {image.shape}")
    return image


def mask_for(row: dict, cache: dict[str, np.ndarray], loader) -> np.ndarray:
    key = row["sample_id"]
    if key not in cache:
        _, mask = loader._load_product(row, 256)
        cache[key] = np.asarray(mask, dtype=np.uint8) > 127
    return cache[key]


def to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device).float() / 255.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--small", type=Path, required=True)
    p.add_argument("--matched", type=Path, required=True)
    p.add_argument("--pix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--limit", type=int, default=2304)
    args = p.parse_args()
    if args.output.exists():
        p.error("--output must be a new path")
    if args.batch_size < 1:
        p.error("batch size must be positive")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo))
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import (
        ExpandedUnifiedProductEditDataset,
    )
    from LightGenV2.tasks.t12_text_to_image.product_object_replace_data import (
        ProductObjectReplacementDataset,
    )

    data = args.assets / "datasets"
    dataset = ExpandedUnifiedProductEditDataset(
        data / "abo_cleanrender_lamp_table_pillow_256_v1", "test", 256,
        data / "abo_unified_expanded_instructions_qwen2_v2.pt",
    )
    if len(dataset) != 2304:
        raise ValueError(f"expected 2304 TEST pairs, got {len(dataset)}")
    if not 1 <= args.limit <= 2304:
        p.error("--limit must be from 1 to 2304")
    models = {
        "small_sim": args.small / "small_sim",
        "small_exp_tuned": args.small / "small_exp",
        "large_sim": args.matched / "large_sim",
        "qwen28_sim": args.matched / "qwen_baseline",
        "pix2pix_turbo_sim": args.pix / "generated",
    }
    target_dir = args.matched / "target"
    for root in [target_dir, *models.values()]:
        count = len(list(root.glob("test_*.png")))
        if count != 2304:
            raise ValueError(f"expected 2304 PNGs under {root}, got {count}")

    device = torch.device("cuda:0")
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True,
    ).to(device).eval()
    mask_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    with torch.no_grad():
        for start in range(0, args.limit, args.batch_size):
            stop = min(start + args.batch_size, args.limit)
            targets = []
            regions = []
            modes = []
            roi_boxes = []
            source_ids = []
            for index in range(start, stop):
                name = f"test_{index:05d}.png"
                target = rgb(target_dir / name)
                source = dataset.sources[index // 12]
                offset = index % 12
                mode = "background" if offset < 4 else "object" if offset < 8 else "joint"
                source_mask = mask_for(source, mask_cache, ProductObjectReplacementDataset)
                design = dataset.by_category[source["category"]][offset % 4][1]
                target_mask = (source_mask if mode == "background" else
                               mask_for(design, mask_cache, ProductObjectReplacementDataset))
                product_union = binary_dilation(source_mask | target_mask, iterations=12)
                y, x = np.nonzero(product_union)
                roi_boxes.append((int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1))
                edit_region = ~source_mask if mode == "background" else (
                    product_union if mode == "object" else np.ones((256, 256), dtype=bool)
                )
                regions.append((edit_region, source_mask if mode == "background" else ~product_union))
                targets.append(target)
                modes.append(mode)
                source_ids.append(source["sample_id"])
            target_tensor = to_tensor(targets, device)
            roi_target = torch.cat([
                F.interpolate(target_tensor[k:k+1, :, y0:y1, x0:x1], size=(256, 256), mode="bilinear", align_corners=False)
                for k, (y0, y1, x0, x1) in enumerate(roi_boxes)
            ])

            for model, directory in models.items():
                predictions = [rgb(directory / f"test_{index:05d}.png") for index in range(start, stop)]
                prediction_tensor = to_tensor(predictions, device)
                full = lpips(prediction_tensor, target_tensor).detach().cpu().flatten().tolist()
                roi_prediction = torch.cat([
                    F.interpolate(prediction_tensor[k:k+1, :, y0:y1, x0:x1], size=(256, 256), mode="bilinear", align_corners=False)
                    for k, (y0, y1, x0, x1) in enumerate(roi_boxes)
                ])
                object_roi = lpips(roi_prediction, roi_target).detach().cpu().flatten().tolist()
                for k, index in enumerate(range(start, stop)):
                    pred_float = predictions[k].astype(np.float32) / 255.0
                    target_float = targets[k].astype(np.float32) / 255.0
                    diff = pred_float - target_float
                    squared = np.square(diff).mean(axis=2)
                    absolute = np.abs(diff).mean(axis=2)
                    edit_mask, preserve_mask = regions[k]
                    mse = float(squared.mean())
                    edit_mse = float(squared[edit_mask].mean())
                    preserve_mae = float(absolute[preserve_mask].mean()) if preserve_mask.any() else None
                    rows.append({
                        "index": index, "sample_id": f"test_{index:05d}", "source_id": source_ids[k],
                        "model": model, "mode": modes[k], "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                        "edit_region_psnr_db": -10.0 * math.log10(max(edit_mse, 1e-12)),
                        "lpips_alex_full": float(full[k]), "lpips_alex_product_roi": float(object_roi[k]),
                        "preserve_region_mae": preserve_mae,
                        "edit_region_fraction": float(edit_mask.mean()),
                    })
            if stop % 120 == 0 or stop == args.limit:
                print(f"processed {stop}/{args.limit}", flush=True)

    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for model in models:
        for mode in ("all", "background", "object", "joint"):
            selected = [r for r in rows if r["model"] == model and (mode == "all" or r["mode"] == mode)]
            summary.append({
                "model": model, "mode": mode, "samples": len(selected),
                **{key: float(np.mean([r[key] for r in selected if r[key] is not None]))
                   for key in ("psnr_db", "edit_region_psnr_db", "lpips_alex_full",
                               "lpips_alex_product_roi", "preserve_region_mae", "edit_region_fraction")},
            })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps([r for r in summary if r["mode"] == "all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
