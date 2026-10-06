"""Join saved metric summaries, without model evaluation or import-time writes."""
import argparse
import csv
from pathlib import Path


def load(root, name):
    with (root / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (row["model"], row["mode"])
        if key in result:
            raise ValueError(f"Duplicate model/mode key in {name}: {key}")
        result[key] = row
    return result


def build_rows(root):
    pixel = load(root, "summary_lpips.csv")
    dists = load(root, "summary_dists.csv")
    regions = load(root, "summary_regions.csv")
    models = ("small_sim", "large_sim", "qwen28_sim", "pix2pix_turbo_sim", "small_exp_tuned")
    modes = ("background", "object", "joint")
    rows = []
    for mode in modes:
        for model in models:
            key = (model, mode)
            a, d, r = pixel[key], dists[key], regions[key]
            if not int(a["samples"]) == int(d["samples"]) == int(r["samples"]) == 768:
                raise ValueError(f"Expected 768 samples for {key}")
            if mode == "background":
                edit_lpips, edit_dists = r["background_masked_lpips_alex"], r["background_masked_dists"]
                edit_method = "neutral-filled product union (12px dilation); background pixels"
                preserve_mae = r["product_mae"]
            elif mode == "object":
                edit_lpips, edit_dists = a["lpips_alex_product_roi"], d["dists_product_roi"]
                edit_method = "dilated source-target union bounding-box ROI, resized 256x256"
                preserve_mae = r["background_mae"]
            else:
                edit_lpips, edit_dists = a["lpips_alex_full"], d["dists_full"]
                edit_method = "full image; no preserved region"
                preserve_mae = ""
            rows.append({
                "model": model, "condition": "EXP tuned checkpoint" if model == "small_exp_tuned" else "clean SIM",
                "mode": mode, "samples": 768,
                "full_psnr_db": a["psnr_db"], "edit_psnr_db": a["edit_region_psnr_db"],
                "edit_lpips_alex": edit_lpips, "edit_dists": edit_dists,
                "edit_perceptual_method": edit_method, "preserved_region_mae": preserve_mae,
                "old_footprint_mae": r["old_only_mae"] if mode != "background" else "",
                "old_footprint_valid_samples": r["old_only_valid_samples"] if mode != "background" else "",
                "outer_boundary_mae": r["boundary_outer_mae"],
                "product_roi_lpips_alex": a["lpips_alex_product_roi"],
                "background_masked_lpips_alex": r["background_masked_lpips_alex"],
            })
    return rows


def write_summary(root, output):
    rows = build_rows(Path(root))
    # Exclusive create also protects against an output appearing after validation.
    with Path(output).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"wrote {write_summary(args.input, args.output)} fixed-TEST mode/model rows")


if __name__ == "__main__":
    main()
