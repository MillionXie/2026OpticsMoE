"""Full-source CLEVR vision pretraining for a frozen-front *candidate* protocol.

The 24-way head is temporary. Only the 128-feature CNN may later be shared by
MoE and D2NN, if that protocol is selected. No optical run uses it implicitly.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t09_multimodal_matching.prepare import COLORS, SHAPES
from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder

from .train_eurosat import batches, save_json, sha256_file


def load_split(root, split):
    images = np.load(root / f"{split}_images.npy", mmap_mode="r")
    image_index = np.load(root / f"{split}_image_index.npy", mmap_mode="r")
    token_ids = np.load(root / f"{split}_token_ids.npy", mmap_mode="r")
    labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
    if (len(labels) != len(images) * 6 or
            not np.array_equal(image_index.reshape(-1, 6),
                               np.broadcast_to(np.arange(len(images))[:, None],
                                               (len(images), 6)))):
        raise ValueError("expected six question records for each image")
    vocab = json.loads((root / "vocab.json").read_text())
    color_ids = np.array([vocab[name] for name in COLORS], dtype=np.uint8)
    shape_ids = np.array([vocab[name] for name in SHAPES], dtype=np.uint8)
    query = np.empty(len(labels), dtype=np.int64)
    for start in range(0, len(labels), 65536):
        part = np.asarray(token_ids[start:start + 65536])
        colors = (part[:, :, None] == color_ids).any(1)
        shapes = (part[:, :, None] == shape_ids).any(1)
        if not np.all(colors.sum(1) == 1) or not np.all(shapes.sum(1) == 1):
            raise ValueError("question must specify one color and shape")
        query[start:start + len(part)] = colors.argmax(1) * len(SHAPES) + shapes.argmax(1)
    return images, query.reshape(-1, 6), np.asarray(labels).reshape(-1, 6)


@torch.no_grad()
def evaluate(model, data, device, batch):
    model.eval()
    images, query, target = data
    loss_sum = 0.0
    correct = 0
    for indices in batches(np.arange(len(images)), batch):
        raw = torch.as_tensor(np.array(images[indices], copy=True), device=device)
        _, logits = model(raw)
        q = torch.as_tensor(query[indices], device=device, dtype=torch.long)
        y = torch.as_tensor(target[indices], device=device, dtype=torch.float32)
        chosen = logits.gather(1, q)
        loss_sum += float(F.binary_cross_entropy_with_logits(chosen, y, reduction="sum"))
        correct += int(((chosen >= 0) == y.bool()).sum())
    records = len(images) * 6
    return {"bce": loss_sum / records, "accuracy": correct / records,
            "images": len(images), "queries": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-images", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not (args.epochs > 0 and args.patience > 0 and args.batch_images > 0 and args.lr > 0):
        raise ValueError("invalid pretraining budget")
    if ("runs", "simulation") not in list(zip(args.out.parts, args.out.parts[1:])):
        raise ValueError("run must live under runs/simulation")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("storage") != "clevr_lazy_v1" or protocol.get("all_original_samples") is not True:
        raise ValueError("expected full original CLEVR source")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    train = load_split(args.protocol.parent, "train")
    val = load_split(args.protocol.parent, "val")
    if len(train[0]) != 70000 or len(val[0]) != 7500:
        raise ValueError("CLEVR image split changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    config = {
        "purpose": "candidate frozen visual front; temporary 24-way head is discarded",
        "source": "full original CLEVR image/query train and validation split",
        "train_images": 70000, "validation_images": 7500,
        "train_queries": 420000, "validation_queries": 45000,
        "protocol_sha256": sha256_file(args.protocol),
        "feature_parameters": sum(p.numel() for p in model.features.parameters()),
        "temporary_head_parameters": sum(p.numel() for p in model.head.parameters()),
        "epochs": args.epochs, "patience": args.patience,
        "batch_images": args.batch_images, "lr": args.lr, "seed": args.seed,
        "selection": "lowest full-validation query BCE", "test_policy": "never read",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device)},
    }
    save_json(args.out / "config.json", config)
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    save_json(args.out / "status.json", {"status": "running", "epoch": 0})
    history = []
    best, best_epoch, wait = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        began = time.time()
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train[0]))
        for indices in batches(order, args.batch_images):
            raw = torch.as_tensor(np.array(train[0][indices], copy=True), device=device)
            q = torch.as_tensor(train[1][indices], device=device, dtype=torch.long)
            y = torch.as_tensor(train[2][indices], device=device, dtype=torch.float32)
            _, logits = model(raw)
            loss = F.binary_cross_entropy_with_logits(logits.gather(1, q), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = evaluate(model, val, device, args.batch_images)
        if validation["bce"] < best - 1e-5:
            best, best_epoch, wait = validation["bce"], epoch, 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "validation": validation},
                       args.out / "best_checkpoint.pt")
        else:
            wait += 1
        history.append({"epoch": epoch, "validation": validation,
                        "seconds": time.time() - began})
        save_json(args.out / "history.json", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "config": config}, args.out / "last_checkpoint.pt")
        save_json(args.out / "status.json", {"status": "running", "epoch": epoch,
                                             "best_epoch": best_epoch, "best_val_bce": best})
        print(json.dumps({"epoch": epoch, "validation": validation,
                          "best_epoch": best_epoch, "seconds": history[-1]["seconds"]}),
              flush=True)
        if wait >= args.patience:
            break
    selected = torch.load(args.out / "best_checkpoint.pt", map_location="cpu",
                          weights_only=False)
    save_json(args.out / "result.json", {"best_epoch": best_epoch,
                                          "validation": selected["validation"],
                                          "test": "not run"})
    save_json(args.out / "status.json", {"status": "complete", "epoch": epoch,
                                         "best_epoch": best_epoch})


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
