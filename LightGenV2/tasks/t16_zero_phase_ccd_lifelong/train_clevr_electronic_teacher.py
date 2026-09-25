"""Train-only image/question teacher for diagnosing original CLEVR matching.

The teacher never participates in MoE or D2NN inference. It uses the same
original paired questions and split, and does not read the test set.
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
from torch import nn
from torch.nn import functional as F

from LightGenV2.tasks.t14_shared_readout_lifelong.data import ClevrRawPairs

from .train_eurosat import batches, save_json, sha256_file
from .train_other_tasks import metrics


class ClevrTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )
        self.tokens = nn.Embedding(26, 32, padding_idx=0)
        self.query = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.condition = nn.Linear(64, 2 * 96)
        self.classifier = nn.Sequential(nn.Linear(2 * 96 + 64, 64), nn.ReLU(),
                                        nn.Linear(64, 2))

    def forward(self, image, tokens):
        visual = self.image(image)
        mask = (tokens != 0).float()
        words = self.tokens(tokens)
        text = self.query((words * mask[:, :, None]).sum(1)
                          / mask.sum(1, keepdim=True).clamp_min(1))
        gamma, beta = self.condition(text).chunk(2, dim=1)
        visual = F.relu(visual * (1 + 0.5 * gamma.tanh()[:, :, None, None])
                        + beta[:, :, None, None])
        pooled = torch.cat((visual.mean((-2, -1)), visual.amax((-2, -1))), 1)
        return self.classifier(torch.cat((pooled, text), 1))


def batch_from(data, indices, device):
    indices = np.asarray(indices, dtype=np.int64)
    raw = np.array(data.images[data.image_index[indices]], copy=True)
    image = torch.as_tensor(raw, device=device).float().permute(0, 3, 1, 2) / 255
    tokens = torch.as_tensor(np.array(data.token_ids[indices], copy=True),
                             device=device, dtype=torch.long)
    if tokens.max() > 25:
        raise ValueError("CLEVR vocabulary changed")
    label = torch.as_tensor(data.labels[indices], device=device, dtype=torch.long)
    return image, tokens, label


@torch.no_grad()
def evaluate(model, data, device, batch):
    model.eval()
    predictions = []
    for indices in batches(np.arange(len(data)), batch):
        image, tokens, _ = batch_from(data, indices, device)
        predictions.append(model(image, tokens).argmax(1).cpu().numpy())
    return metrics(data.labels, np.concatenate(predictions), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not (args.epochs > 0 and args.patience > 0 and args.batch > 0 and args.lr > 0):
        raise ValueError("invalid training budget")
    if ("runs", "simulation") not in list(zip(args.out.parts, args.out.parts[1:])):
        raise ValueError("run must live under runs/simulation")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("storage") != "clevr_lazy_v1" or protocol.get("all_original_samples") is not True:
        raise ValueError("expected original full CLEVR source")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    train = ClevrRawPairs(args.protocol.parent, "train")
    val = ClevrRawPairs(args.protocol.parent, "val")
    if len(train) != 140000 or len(val) != 15000:
        raise ValueError("CLEVR split changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClevrTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    config = {
        "purpose": "train-only electronic teacher; not an optical inference result",
        "task": "original CLEVR image plus original positive/negative question",
        "input": "raw 64x64 RGB and 32 original token IDs",
        "output_classes": 2, "test_policy": "never read",
        "train_records": len(train), "validation_records": len(val),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "protocol_sha256": sha256_file(args.protocol),
        "epochs": args.epochs, "patience": args.patience,
        "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device)},
    }
    save_json(args.out / "config.json", config)
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    save_json(args.out / "status.json", {"status": "running", "epoch": 0})
    history = []
    best, best_epoch, wait = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        began = time.time()
        model.train()
        loss_sum = 0.0
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        for indices in batches(order, args.batch):
            image, tokens, target = batch_from(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image, tokens), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        validation = evaluate(model, val, device, args.batch)
        score = validation["balanced_accuracy"]
        if score > best + 1e-5:
            best, best_epoch, wait = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "validation": validation}, args.out / "best_checkpoint.pt")
        else:
            wait += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train),
                        "validation": validation, "seconds": time.time() - began})
        save_json(args.out / "history.json", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "config": config}, args.out / "last_checkpoint.pt")
        save_json(args.out / "status.json", {"status": "running", "epoch": epoch,
                                             "best_epoch": best_epoch, "best_val": best})
        print(json.dumps({"epoch": epoch, "val": score, "best": best,
                          "seconds": history[-1]["seconds"]}), flush=True)
        if wait >= args.patience:
            break
    save_json(args.out / "result.json", {"best_epoch": best_epoch,
                                          "validation": torch.load(
                                              args.out / "best_checkpoint.pt",
                                              map_location="cpu", weights_only=False
                                          )["validation"], "test": "not run"})
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
