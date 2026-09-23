"""First admission experiment: raw EuroSAT into optics with a fixed shared head.

This command trains only optical parameters. It never loads the previously
label-trained EuroSAT CNN, and it chooses the checkpoint on validation only.
The test split is evaluated once after the checkpoint has been selected.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from LightGenV2.tasks.t13_four_modal_lifelong.data import EuroSatFields
from .model import SharedReadoutOptics

EXPECTED_SIZES = {"train": 31996, "val": 10930, "test": 10858}


def source_paths(protocol_path: Path):
    protocol = json.loads(protocol_path.read_text())
    source = protocol.get("source_protocol", protocol)
    if source.get("storage") != "eurosat_rgb_sar_v1":
        raise ValueError("expected the audited raw RGB/SAR source protocol")
    if source.get("all_original_samples") is not True:
        raise ValueError("source protocol does not certify the full source split")
    return Path(source["trainval_npz"]), Path(source["holdout_npz"])


def split_data(trainval: Path, holdout: Path, split: str):
    fields = EuroSatFields(trainval, holdout, split)
    data = np.load(holdout if split == "test" else trainval,
                   mmap_mode="r", allow_pickle=False)
    prefix = "validation" if split == "val" else split
    labels = np.asarray(data[prefix + "_labels"], dtype=np.int64)
    domains = np.asarray(data[prefix + "_domains"], dtype=np.int64)
    assert len(fields) == len(labels) == len(domains)
    assert set(np.unique(labels)) == set(range(10))
    return fields, labels, domains


def batches(indices, size):
    for begin in range(0, len(indices), size):
        yield indices[begin:begin + size]


@torch.no_grad()
def evaluate(model, split, device, batch_size, max_batches=None):
    fields, labels, domains = split
    model.eval()
    predictions = []
    for step, ix in enumerate(batches(np.arange(len(labels)), batch_size)):
        if max_batches is not None and step >= max_batches:
            break
        output = model(fields[ix].to(device))["logits"]
        predictions.append(output.argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    labels = labels[:len(predicted)]
    domains = domains[:len(predicted)]
    result = {"accuracy": float(accuracy_score(labels, predicted)),
              "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
              "n": int(len(labels))}
    result["domains"] = {
        str(int(domain)): {
            "accuracy": float(accuracy_score(labels[domains == domain],
                                              predicted[domains == domain])),
            "balanced_accuracy": float(balanced_accuracy_score(
                labels[domains == domain], predicted[domains == domain])),
            "n": int((domains == domain).sum())}
        for domain in np.unique(domains)}
    return result


def state_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--architecture", choices=("moe", "d2nn"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--readout-gain", type=float, default=1.0)
    parser.add_argument("--readout-design", choices=("orthogonal", "windows"),
                        default="orthogonal")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-test", action="store_true",
                        help="validation-only candidate; never open holdout")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="optical-only fine-tune from a validation-selected checkpoint")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="debug only: output marked non-formal and test is skipped")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="debug only: output marked non-formal and test is skipped")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainval, holdout = source_paths(args.protocol)
    # Test tensors are not even opened until validation has fixed a checkpoint.
    data = {name: split_data(trainval, holdout, name)
            for name in ("train", "val")}
    for name, (_, labels, _) in data.items():
        if len(labels) != EXPECTED_SIZES[name]:
            raise ValueError(f"{name} split has {len(labels)}, expected {EXPECTED_SIZES[name]}")
    model = SharedReadoutOptics(args.architecture, seed=args.seed,
                                max_experts=16,
                                readout_gain=args.readout_gain,
                                readout_design=args.readout_design).to(device)
    model.configure_stage(0)
    head_hash = state_hash(model.shared_head.weight)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device,
                             weights_only=False)
        if initial["head_sha256"] != head_hash:
            raise ValueError("initial checkpoint uses a different frozen readout")
        model.load_state_dict(initial["model"])
        if state_hash(model.shared_head.weight) != head_hash:
            raise ValueError("loading changed the fixed readout")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr)
    formal = args.max_train_batches is None and args.max_eval_batches is None
    metadata = {"contract": "raw_RGB_or_SAR_to_optics; one fixed shared Linear(784,10); optical training only",
                "architecture": args.architecture, "seed": args.seed,
                "source_protocol": str(args.protocol), "sizes": {k: len(v[1]) for k, v in data.items()},
                "head_sha256": head_hash, "formal_full_train": formal,
                "readout_gain": args.readout_gain,
                "readout_design": args.readout_design,
                "epochs": args.epochs, "min_epochs": args.min_epochs,
                "patience": args.patience, "batch": args.batch,
                "no_label_trained_frontend": True}
    metadata["init_checkpoint"] = (str(args.init_checkpoint)
                                   if args.init_checkpoint is not None else None)
    metadata["test_will_be_evaluated"] = bool(formal and not args.skip_test)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best, best_epoch, wait = -1.0, None, 0
    history = []
    if args.init_checkpoint is not None:
        initial_validation = evaluate(model, data["val"], device,
                                      args.eval_batch, args.max_eval_batches)
        best, best_epoch = initial_validation["balanced_accuracy"], 0
        history.append({"epoch": 0, "train_loss": None,
                        "validation": initial_validation, "seconds": 0.0})
        torch.save({"model": model.state_dict(), "epoch": 0,
                    "validation": initial_validation, "head_sha256": head_hash},
                   args.out / "best_checkpoint.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    fields, labels, _ = data["train"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        began = time.time()
        order = np.random.default_rng(args.seed + epoch).permutation(len(labels))
        losses = []
        for step, ix in enumerate(batches(order, args.batch)):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            x = fields[ix].to(device)
            y = torch.as_tensor(labels[ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = F.cross_entropy(output["logits"], y)
            if args.architecture == "moe":
                q = output["route_power"][:, :4]
                loss = loss + 0.1 * (q.mean(0) - 0.25).square().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        assert state_hash(model.shared_head.weight) == head_hash
        validation = evaluate(model, data["val"], device, args.eval_batch,
                              args.max_eval_batches)
        score = validation["balanced_accuracy"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation": validation, "seconds": time.time() - began}
        history.append(row)
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "val_balanced_accuracy": score,
                          "seconds": row["seconds"]}), flush=True)
        if score > best + 1e-5:
            best, best_epoch, wait = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation": validation, "head_sha256": head_hash},
                       args.out / "best_checkpoint.pt")
        else:
            wait += 1
        if epoch >= args.min_epochs and wait >= args.patience:
            break
    result = {"selected_epoch": best_epoch, "validation": best,
              "head_sha256_unchanged": state_hash(model.shared_head.weight) == head_hash,
              "formal_full_train": formal}
    if formal and not args.skip_test:
        selected = torch.load(args.out / "best_checkpoint.pt", map_location=device,
                              weights_only=False)
        model.load_state_dict(selected["model"])
        test = split_data(trainval, holdout, "test")
        if len(test[1]) != EXPECTED_SIZES["test"]:
            raise ValueError("test split size does not match the audited source")
        result["test"] = evaluate(model, test, device, args.eval_batch)
        metadata["sizes"]["test"] = len(test[1])
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
