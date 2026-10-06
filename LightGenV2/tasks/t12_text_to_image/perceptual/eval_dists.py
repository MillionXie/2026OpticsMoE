"""Independent DISTS cross-check on the same 2304 paired TEST PNGs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from DISTS_pytorch import DISTS

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
    roots = {
        "small_sim": args.small / "small_sim",
        "small_exp_tuned": args.small / "small_exp",
        "large_sim": args.matched / "large_sim",
        "qwen28_sim": args.matched / "qwen_baseline",
        "pix2pix_turbo_sim": args.pix / "generated",
    }
    target_dir = args.matched / "target"
    for root in [target_dir, *roots.values()]:
        if len(list(root.glob("test_*.png"))) != 2304:
            raise ValueError(f"Incomplete image series: {root}")

    device = torch.device("cuda:0")
    metric = DISTS(load_weights=False)
    weights = torch.load(args.dists_weights, map_location="cpu", weights_only=False)
    metric.alpha.data.copy_(weights["alpha"])
    metric.beta.data.copy_(weights["beta"])
    metric.to(device).eval()
    mask_cache: dict[str, np.ndarray] = {}
    rows = []
    with torch.no_grad():
        for start in range(0, 2304, args.batch_size):
            stop = min(start + args.batch_size, 2304)
            targets = [rgb(target_dir / f"test_{i:05d}.png") for i in range(start, stop)]
            target_tensor = to_tensor(targets, device)
            boxes = []
            modes = []
            for i in range(start, stop):
                source = dataset.sources[i // 12]
                offset = i % 12
                mode = "background" if offset < 4 else "object" if offset < 8 else "joint"
                design = dataset.by_category[source["category"]][offset % 4][1]
                a = mask_for(source, mask_cache, ProductObjectReplacementDataset)
                b = mask_for(design, mask_cache, ProductObjectReplacementDataset)
                from scipy.ndimage import binary_dilation
                mask = binary_dilation(a | b, iterations=12)
                y, x = np.nonzero(mask)
                boxes.append((int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1))
                modes.append(mode)
            target_roi = torch.cat([
                F.interpolate(target_tensor[k:k+1, :, y0:y1, x0:x1], size=(256, 256), mode="bilinear", align_corners=False)
                for k, (y0, y1, x0, x1) in enumerate(boxes)
            ])
            for name, root in roots.items():
                pred = to_tensor([rgb(root / f"test_{i:05d}.png") for i in range(start, stop)], device)
                pred_roi = torch.cat([
                    F.interpolate(pred[k:k+1, :, y0:y1, x0:x1], size=(256, 256), mode="bilinear", align_corners=False)
                    for k, (y0, y1, x0, x1) in enumerate(boxes)
                ])
                full = metric(pred, target_tensor).detach().cpu().flatten().tolist()
                roi = metric(pred_roi, target_roi).detach().cpu().flatten().tolist()
                for k, i in enumerate(range(start, stop)):
                    rows.append({"index": i, "model": name, "mode": modes[k],
                                 "dists_full": float(full[k]), "dists_product_roi": float(roi[k])})
            if stop % 120 == 0 or stop == 2304:
                print(f"processed {stop}/2304", flush=True)

    with (args.output / "per_image_dists.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for name in roots:
        for mode in ("all", "background", "object", "joint"):
            sample = [r for r in rows if r["model"] == name and (mode == "all" or r["mode"] == mode)]
            summary.append({"model": name, "mode": mode, "samples": len(sample),
                            "dists_full": float(np.mean([r["dists_full"] for r in sample])),
                            "dists_product_roi": float(np.mean([r["dists_product_roi"] for r in sample]))})
    (args.output / "summary_dists.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary_dists.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(json.dumps([r for r in summary if r["mode"] == "all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
