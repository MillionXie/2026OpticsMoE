"""Train the shared Physical Concepts video encoder before optical matching."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PhysicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d((28, 28)), nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.GELU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        x = x.float()
        # Optical amplitude fields have unit total power and therefore values
        # around 1/224. Per-sample mean scaling prevents the MLP biases from
        # dominating these small but informative temporal tiles.
        x = x / x.mean((-2, -1), keepdim=True).clamp_min(1e-6)
        z = self.features(x[:, None])
        return z, self.head(z)


@torch.no_grad()
def evaluate(model, fields, labels, device, batch):
    model.eval(); prediction = []
    for i in range(0, len(labels), batch):
        x = torch.as_tensor(np.array(fields[i:i+batch], copy=True), device=device)
        prediction.append(model(x)[1].argmax(1).cpu().numpy())
    return float((np.concatenate(prediction) == labels).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17); np.random.seed(17)
    data = {}
    for split in ("train", "val", "test"):
        z = np.load(a.data / f"{split}.npz", mmap_mode="r", allow_pickle=False)
        data[split] = (z["fields"], np.asarray(z["labels"], dtype=np.int64))
    model = PhysicalEncoder().to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs, eta_min=1e-4)
    best = -1.0; history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); order = np.random.default_rng(17 + epoch).permutation(len(data["train"][1]))
        losses = []
        for start in range(0, len(order), a.batch):
            ix = order[start:start+a.batch]
            x = torch.as_tensor(np.array(data["train"][0][ix], copy=True), device=a.device)
            y = torch.as_tensor(data["train"][1][ix], device=a.device)
            loss = F.cross_entropy(model(x)[1], y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "train_accuracy": evaluate(model, *data["train"], a.device, a.batch),
               "val_accuracy": evaluate(model, *data["val"], a.device, a.batch)}
        history.append(row); (a.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "validation": row["val_accuracy"]}
        torch.save(checkpoint, a.out / "last_checkpoint.pt")
        if row["val_accuracy"] > best:
            best = row["val_accuracy"]; torch.save(checkpoint, a.out / "best_checkpoint.pt")
        print(json.dumps(row), flush=True)
    checkpoint = torch.load(a.out / "best_checkpoint.pt", map_location=a.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"selected_epoch": checkpoint["epoch"], "validation_accuracy": checkpoint["validation"],
              "test_accuracy": evaluate(model, *data["test"], a.device, a.batch),
              "retained_feature_parameters": sum(p.numel() for p in model.features.parameters()),
              "temporary_head_parameters": sum(p.numel() for p in model.head.parameters())}
    (a.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
