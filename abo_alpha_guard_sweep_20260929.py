"""Select four existing in-block optical fractions on TRAIN continuation holdout.

No new branch or optical phase change. Original 800 TEST queries are evaluated
once after the four alpha scalars have been selected.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from transformers import AutoProcessor

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256, verify_assets, write_json
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval, alpha_value
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import assessment, fitting_groups
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import load_screen
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.robust_holdout import split_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--noise-metadata-checkpoint", type=Path, required=True)
    parser.add_argument("--noise-metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--targets", default="original,.31,.34,.37,.40")
    parser.add_argument("--clean-floor", type=float, default=.975)
    args = parser.parse_args()
    targets = tuple("original" if item.strip() == "original" else float(item)
                    for item in args.targets.split(","))
    if not targets or len(set(targets)) != len(targets) or "original" not in targets:
        raise ValueError("Targets must be unique and include original")
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing sweep")
    if sha256(args.checkpoint) != args.checkpoint_sha256 or sha256(args.noise_metadata_checkpoint) != args.noise_metadata_sha256:
        raise ValueError("Checkpoint SHA mismatch")
    verify_assets(args.assets)
    protocol, groups = load_screen(args.manifest, args.data)
    reduced, holdout, audit = split_train(groups)
    fit_holdout = fitting_groups(protocol, reduced, multi_view=True)
    fit_full = fitting_groups(protocol, groups, multi_view=True)
    args._selection_groups = holdout
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    noise = torch.load(args.noise_metadata_checkpoint, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    model = OpticalRetrieval(copy.deepcopy(noise["metadata"])).to(device)
    model.load_state_dict(source["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.assets / "processor"), local_files_only=True)
    keys = ("vision.block1_optical_fusion_logit", "vision.block2_optical_fusion_logit",
            "language.block1_optical_fusion_logit", "language.block2_optical_fusion_logit")
    original = {key: source["state_dict"][key].detach().clone() for key in keys}
    args.output.mkdir(parents=True)
    write_json(args.output / "execution.json", {
        "checkpoint_sha256": args.checkpoint_sha256, "noise_metadata_sha256": args.noise_metadata_sha256,
        "changed_keys_only": keys, "target_alphas": targets, "clean_floor": args.clean_floor,
        "selection": "400 continuation TRAIN holdout clean/noisy only; original TEST after selection",
        "holdout_audit": audit, "caveat": "Continuation holdout seen by original checkpoint earlier; not independent",
    })
    rows = []
    selected = None
    for value in targets:
        with torch.no_grad():
            for key in keys:
                p = dict(model.named_parameters())[key]
                if value == "original":
                    p.copy_(original[key].to(device))
                else:
                    modality = model.vision if key.startswith("vision.") else model.language
                    low, high = modality.alpha_bounds
                    if not low < value < high:
                        raise ValueError(f"Target alpha outside bounds: {value}, {low}, {high}")
                    p.fill_(math.log((value - low) / (high - value)))
        measured = assessment(model, processor, groups, args, device, fit=fit_holdout, selection=True)
        clean = float(measured["test"]["r_at_1"])
        noisy = float(measured["validation_noisy"]["r_at_1"])
        fractions = {key: float(alpha_value(dict(model.named_parameters())[key],
                        (model.vision if key.startswith("vision.") else model.language).alpha_bounds)) for key in keys}
        row = {"target": value, "alphas": fractions, "holdout_clean_r1": clean,
               "holdout_noisy_r1": noisy, "eligible": clean >= args.clean_floor,
               "vision_router": measured["router"]["vision"]["selection_share"],
               "language_router": measured["router"]["language"]["selection_share"]}
        rows.append(row)
        write_json(args.output / "validation.json", rows)
        print(json.dumps(row), flush=True)
        if row["eligible"] and (selected is None or noisy > selected["holdout_noisy_r1"]):
            selected = row
    if selected is None:
        write_json(args.output / "report.json", {"status": "no_clean_eligible_target",
              "validation": rows, "clean_floor": args.clean_floor})
        return
    with torch.no_grad():
        for key in keys:
            p = dict(model.named_parameters())[key]
            if selected["target"] == "original":
                p.copy_(original[key].to(device))
            else:
                modality = model.vision if key.startswith("vision.") else model.language
                low, high = modality.alpha_bounds
                value = selected["target"]
                p.fill_(math.log((value - low) / (high - value)))
    output_payload = copy.deepcopy(source)
    output_payload["state_dict"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output_payload["test_selected"] = False
    output_payload["selection_scheme"] = "four_alpha_scalars_train_holdout_v1"
    torch.save(output_payload, args.output / "best.pt")
    write_json(args.output / "report.json", {"status": "validation_selected",
          "selected": selected, "validation": rows, "best_sha256": sha256(args.output / "best.pt")})
    args._selection_groups = None
    normal = assessment(model, processor, groups, args, device, fit=fit_full)
    model.set_remove_optical(True)
    removed = assessment(model, processor, groups, args, device, fit=fit_full)
    model.set_remove_optical(False)
    write_json(args.output / "report.json", {"status": "complete", "selected": selected,
          "validation": rows, "normal": normal, "remove_optical": removed,
          "best_sha256": sha256(args.output / "best.pt"), "phase_unchanged": True,
          "note": "Original TEST once after TRAIN continuation holdout; previously exposed development TEST"})


if __name__ == "__main__":
    main()
