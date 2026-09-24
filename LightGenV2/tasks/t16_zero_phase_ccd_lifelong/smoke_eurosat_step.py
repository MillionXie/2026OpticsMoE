"""One real paired-EuroSAT optimizer step to validate the new readout contract.

This is an engineering smoke check, never a training result or accuracy claim.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import PairedEuroSatFields
from .model import DirectCCDOptics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--architecture", choices=("moe", "d2nn"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.batch < 2 or tuple(args.out.parts[-2:]) == ("runs", "smoke"):
        raise ValueError("batch must be >=2 and --out must name a run below runs/smoke")
    if ("runs", "smoke") not in list(zip(args.out.parts, args.out.parts[1:])):
        raise ValueError("smoke run must live under runs/smoke")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "status.json").write_text('{"status": "running"}\n')
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    protocol = json.loads(args.protocol.read_text())
    source = protocol.get("source_protocol", protocol)
    if source.get("storage") != "eurosat_rgb_sar_v1":
        raise ValueError("expected paired EuroSAT RGB/SAR source")
    data = PairedEuroSatFields(Path(source["trainval_npz"]),
                               Path(source["holdout_npz"]), "train")
    if len(data) != 15998:
        raise ValueError(f"expected 15998 paired train locations, found {len(data)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    model = DirectCCDOptics(args.architecture).to(device)
    indices = np.linspace(0, len(data) - 1, args.batch, dtype=np.int64)
    amplitude = data[indices].to(device)
    labels = torch.as_tensor(data.labels[indices], device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-3)
    before = model(amplitude)
    loss = F.cross_entropy(before["logits"], labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    head_grad = float(model.shared_head.weight.grad.norm())
    phase_grad = float(model.global_phase.grad.norm())
    if not np.isfinite([float(loss), head_grad, phase_grad]).all():
        raise RuntimeError("nonfinite loss or gradient")
    if head_grad <= 0 or phase_grad <= 0:
        raise RuntimeError("CCD head or optical phase received no gradient")
    head_before = model.shared_head.weight.detach().clone()
    phase_before = model.global_phase.detach().clone()
    optimizer.step()
    with torch.no_grad():
        after = F.cross_entropy(model(amplitude)["logits"], labels)
    report = {
        "formal": False,
        "accuracy_claim": False,
        "architecture": args.architecture,
        "activation_order": model.activation_order,
        "train_split_locations": len(data),
        "smoke_batch": args.batch,
        "initial_loss": float(loss),
        "after_one_step_loss": float(after),
        "head_grad_norm": head_grad,
        "global_phase_grad_norm": phase_grad,
        "head_updated": bool(torch.any(model.shared_head.weight != head_before)),
        "global_phase_updated": bool(torch.any(model.global_phase != phase_before)),
        "model_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "device": str(device)},
        "source_protocol": str(args.protocol),
    }
    if not np.isfinite(float(after)) or not report["head_updated"] or not report["global_phase_updated"]:
        raise RuntimeError("optimizer step did not update both optical and electronic parameters")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "status.json").write_text('{"status": "complete"}\n')
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
