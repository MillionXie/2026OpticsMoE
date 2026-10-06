"""Mode-aware edited and preserved-region metrics for fixed T12 TEST PNGs.

Ground-truth alpha masks are used for evaluation only. No model receives masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from DISTS_pytorch import DISTS
from scipy.ndimage import binary_dilation
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from LightGenV2.tasks.t12_text_to_image.perceptual.eval_lpips import mask_for, rgb, to_tensor


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("repo", "assets", "small", "matched", "pix", "output", "dists-weights"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        p.error("--output must not exist")
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
        raise ValueError("Expected 2304 TEST pairs")
    models = {
        "small_sim": args.small / "small_sim",
        "small_exp_tuned": args.small / "small_exp",
        "large_sim": args.matched / "large_sim",
        "qwen28_sim": args.matched / "qwen_baseline",
        "pix2pix_turbo_sim": args.pix / "generated",
    }
    target_dir = args.matched / "target"
    for root in [target_dir, *models.values()]:
        if len(list(root.glob("test_*.png"))) != 2304:
            raise ValueError(f"Incomplete image series: {root}")

    device = torch.device("cuda:0")
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True,
    ).to(device).eval()
    dists = DISTS(load_weights=False)
    weights = torch.load(args.dists_weights, map_location="cpu", weights_only=False)
    dists.alpha.data.copy_(weights["alpha"])
    dists.beta.data.copy_(weights["beta"])
    dists.to(device).eval()
    mask_cache: dict[str, np.ndarray] = {}
    rows = []
    with torch.no_grad():
        for start in range(0, 2304, args.batch_size):
            stop = min(start + args.batch_size, 2304)
            targets = [rgb(target_dir / f"test_{i:05d}.png") for i in range(start, stop)]
            target_tensor = to_tensor(targets, device)
            records = []
            for i in range(start, stop):
                source = dataset.sources[i // 12]
                offset = i % 12
                mode = "background" if offset < 4 else "object" if offset < 8 else "joint"
                design = dataset.by_category[source["category"]][offset % 4][1]
                source_mask = mask_for(source, mask_cache, ProductObjectReplacementDataset)
                target_mask = (source_mask if mode == "background" else
                               mask_for(design, mask_cache, ProductObjectReplacementDataset))
                union = source_mask | target_mask
                protected_product = binary_dilation(union, iterations=12)
                background = ~protected_product
                old_only = source_mask & ~binary_dilation(target_mask, iterations=4)
                boundary = binary_dilation(union, iterations=12) & ~union
                records.append((mode, target_mask, background, old_only, boundary))
            background_tensor = torch.from_numpy(np.stack([r[2] for r in records])).to(device)
            neutral_target = target_tensor.clone()
            neutral_target.masked_fill_(~background_tensor[:, None], 0.5)

            for model, root in models.items():
                predictions = [rgb(root / f"test_{i:05d}.png") for i in range(start, stop)]
                prediction_tensor = to_tensor(predictions, device)
                neutral_prediction = prediction_tensor.clone()
                neutral_prediction.masked_fill_(~background_tensor[:, None], 0.5)
                background_lpips = lpips(neutral_prediction, neutral_target).detach().cpu().flatten().tolist()
                background_dists = dists(neutral_prediction, neutral_target).detach().cpu().flatten().tolist()
                for k, index in enumerate(range(start, stop)):
                    mode, product, background, old_only, boundary = records[k]
                    difference = np.abs(predictions[k].astype(np.float32) - targets[k].astype(np.float32)).mean(axis=2) / 255.0
                    rows.append({
                        "index": index, "model": model, "mode": mode,
                        "background_masked_lpips_alex": float(background_lpips[k]),
                        "background_masked_dists": float(background_dists[k]),
                        "background_mae": float(difference[background].mean()),
                        "product_mae": float(difference[product].mean()),
                        "old_only_mae": float(difference[old_only].mean()) if old_only.any() else None,
                        "old_only_fraction": float(old_only.mean()),
                        "boundary_outer_mae": float(difference[boundary].mean()),
                        "background_fraction": float(background.mean()),
                    })
            if stop % 120 == 0 or stop == 2304:
                print(f"processed {stop}/2304", flush=True)

    with (args.output / "per_image_regions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def mean_or_none(values: list[float | None]) -> float | None:
        valid = [value for value in values if value is not None]
        return float(np.mean(valid)) if valid else None

    summary = []
    for model in models:
        for mode in ("all", "background", "object", "joint"):
            selected = [r for r in rows if r["model"] == model and (mode == "all" or r["mode"] == mode)]
            summary.append({
                "model": model, "mode": mode, "samples": len(selected),
                **{key: mean_or_none([r[key] for r in selected])
                   for key in ("background_masked_lpips_alex", "background_masked_dists",
                               "background_mae", "product_mae", "old_only_mae",
                               "old_only_fraction", "boundary_outer_mae", "background_fraction")},
                "old_only_valid_samples": sum(r["old_only_mae"] is not None for r in selected),
            })
    (args.output / "summary_regions.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "summary_regions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps([r for r in summary if r["mode"] in ("background", "object", "joint")], indent=2), flush=True)


if __name__ == "__main__":
    main()
