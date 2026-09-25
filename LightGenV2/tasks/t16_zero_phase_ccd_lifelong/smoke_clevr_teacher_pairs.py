"""Check whether a conditional image/question teacher can memorize tiny pairs."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t14_shared_readout_lifelong.data import ClevrRawPairs

from .train_clevr_electronic_teacher import ClevrTeacher, batch_from
from .train_eurosat import save_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    if args.pairs <= 0 or args.steps <= 0 or ("runs", "smoke") not in list(zip(
            args.out.parts, args.out.parts[1:])):
        raise ValueError("invalid smoke budget or output path")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ClevrRawPairs(args.protocol.parent, "train")
    indices = np.arange(2 * args.pairs)
    if not np.all(data.labels[indices].reshape(-1, 2) == [1, 0]):
        raise ValueError("expected same-image positive/negative pairs")
    image, tokens, label = batch_from(data, indices, device)
    model = ClevrTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    config = {"purpose": "tiny training-only memorization diagnostic",
              "pairs": args.pairs, "steps": args.steps, "seed": 17,
              "protocol_sha256": sha256_file(args.protocol),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                     text=True).strip(),
              "environment": {"python": platform.python_version(),
                              "torch": torch.__version__, "cuda": torch.version.cuda}}
    save_json(args.out / "config.json", config)
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    save_json(args.out / "status.json", {"status": "running"})
    with torch.no_grad():
        initial = model(image, tokens)
        initial_correct = int((initial.argmax(1) == label).sum())
    for step in range(args.steps):
        model.train()
        logits = model(image, tokens)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = model(image, tokens)
        final_correct = int((final.argmax(1) == label).sum())
        final_loss = float(F.cross_entropy(final, label))
    result = {"training_records": len(indices), "initial_correct": initial_correct,
              "final_correct": final_correct, "final_loss": final_loss,
              "test_or_validation_used": False}
    save_json(args.out / "result.json", result)
    save_json(args.out / "status.json", {"status": "complete"})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            if out.is_dir():
                save_json(out / "status.json", {"status": "failed",
                                                "error": f"{type(error).__name__}: {error}"})
        raise
