"""TRAIN-holdout selection of a phase-preserving electronic weight bridge.

No new branch or changed optical phase. The original 800 TEST queries are
evaluated once after choosing a bridge fraction on 400 continuation TRAIN
holdout queries. That holdout was seen by the starting checkpoint earlier.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import (
    sha256, verify_assets, write_json,
)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import (
    assessment, fitting_groups,
)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import load_screen
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.robust_holdout import split_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--robust", type=Path, required=True)
    parser.add_argument("--robust-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing bridge run")
    if sha256(args.base) != args.base_sha256 or sha256(args.robust) != args.robust_sha256:
        raise ValueError("Input checkpoint SHA mismatch")
    verify_assets(args.assets)
    protocol, groups = load_screen(args.manifest, args.data)
    reduced, holdout, audit = split_train(groups)
    fit_holdout = fitting_groups(protocol, reduced, multi_view=True)
    fit_full = fitting_groups(protocol, groups, multi_view=True)
    args._selection_groups = holdout
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    robust = torch.load(args.robust, map_location="cpu", weights_only=True)
    left, right = base["state_dict"], robust["state_dict"]
    if set(left) != set(right):
        raise ValueError("Checkpoint state keys differ")
    fixed = []
    for key in left:
        if left[key].shape != right[key].shape:
            raise ValueError(f"State shape mismatch: {key}")
        if "optics.experts." in key or key.endswith("optics.global_phase") or key.endswith("raw_router_phase"):
            if not torch.equal(left[key], right[key]):
                raise ValueError(f"Optical phase changed: {key}")
            fixed.append(key)
    if len(fixed) != 12:
        raise ValueError(f"Expected 12 identical phase tensors, got {len(fixed)}")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    model = OpticalRetrieval(copy.deepcopy(robust["metadata"]))
    model.to(device).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.assets / "processor"), local_files_only=True)
    args.output.mkdir(parents=True)
    write_json(args.output / "execution.json", {
        "base_sha256": args.base_sha256, "robust_sha256": args.robust_sha256,
        "manifest_sha256": sha256(args.manifest), "phase_tensors_identical": fixed,
        "selection": "400 continuation TRAIN holdout only; original TEST after selection",
        "holdout_audit": audit, "fractions": [0., .25, .5, .75, 1.],
        "holdout_caveat": "This continuation TRAIN holdout was seen by the starting checkpoint; not independent generalization",
    })
    rows = []
    selected = None
    for fraction in (0., .25, .5, .75, 1.):
        mixed = {}
        for key, tensor in left.items():
            other = right[key]
            if torch.is_floating_point(tensor):
                mixed[key] = torch.lerp(tensor.float(), other.float(), fraction).to(tensor.dtype)
            else:
                if not torch.equal(tensor, other):
                    raise ValueError(f"Nonfloat state differs: {key}")
                mixed[key] = tensor
        model.load_state_dict(mixed, strict=True)
        metrics = assessment(model, processor, groups, args, device, fit=fit_holdout, selection=True)
        clean = float(metrics["test"]["r_at_1"])
        noisy = float(metrics["validation_noisy"]["r_at_1"])
        record = {"fraction": fraction, "holdout_clean_r1": clean,
                  "holdout_noisy_r1": noisy, "eligible": clean >= .975,
                  "vision_router": metrics["router"]["vision"]["selection_share"],
                  "language_router": metrics["router"]["language"]["selection_share"]}
        rows.append(record)
        write_json(args.output / "validation.json", rows)
        print(json.dumps(record), flush=True)
        if record["eligible"] and (selected is None or noisy > selected["holdout_noisy_r1"]):
            selected = record
    if selected is None:
        selected = rows[0]
    fraction = selected["fraction"]
    mixed = {key: (torch.lerp(left[key].float(), right[key].float(), fraction).to(left[key].dtype)
                   if torch.is_floating_point(left[key]) else left[key]) for key in left}
    model.load_state_dict(mixed, strict=True)
    args._selection_groups = None
    normal = assessment(model, processor, groups, args, device, fit=fit_full)
    model.set_remove_optical(True)
    removed = assessment(model, processor, groups, args, device, fit=fit_full)
    model.set_remove_optical(False)
    selected_payload = copy.deepcopy(robust)
    selected_payload["state_dict"] = mixed
    selected_payload["selection_scheme"] = "400_continuation_train_holdout_bridge_v1"
    selected_payload["test_selected"] = False
    selected_payload["bridge_fraction"] = fraction
    torch.save(selected_payload, args.output / "best.pt")
    report = {"status": "complete", "selected_fraction": fraction,
              "selected_validation": selected, "validation": rows,
              "normal": normal, "remove_optical": removed,
              "best_sha256": sha256(args.output / "best.pt"),
              "phase_tensors_identical": len(fixed),
              "note": "Clean TEST evaluated after TRAIN-holdout selection only; previously exposed development TEST and continuation holdout, not independent"}
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
