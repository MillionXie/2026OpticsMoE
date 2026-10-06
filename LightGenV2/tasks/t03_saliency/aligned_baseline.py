"""Frozen Qwen baseline with the SAME adapter specification and density decoder."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher
from experiments.vision2_hybrid_dense.modeling import SaliencyDensityDecoder
from .modeling import load_vision_backbone, sha256_file
from .settings import load_settings, save_resolved_config
from .run import _seed
from .training import _write_json, _write_csv, staged_epoch
from .training_support import use_spawn_workers


class AlignedReadout(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.input_adapter = nn.Linear(hidden_size, 192)
        self.input_norm = nn.LayerNorm(192)
        self.decoder = SaliencyDensityDecoder(192, 224)

    def forward(self, spatial):
        tokens = spatial.float().permute(0, 2, 3, 1)
        tokens = self.input_norm(self.input_adapter(tokens))
        return self.decoder(tokens.permute(0, 3, 1, 2))

    def parameter_audit(self):
        return {"adapter": sum(p.numel() for m in (self.input_adapter, self.input_norm) for p in m.parameters()),
                "decoder": sum(p.numel() for p in self.decoder.parameters()),
                "total": sum(p.numel() for p in self.parameters())}


def _resolve_baseline_protocol(settings, fixed_learning_rate=None, test_interval=None):
    """Preserve defaults; historical fixed-LR snapshot requires explicit opt-in."""
    if fixed_learning_rate is not None and (
        not math.isfinite(fixed_learning_rate) or fixed_learning_rate <= 0
    ):
        raise ValueError("fixed learning rate must be positive and finite")
    interval = settings.test_interval_epochs if test_interval is None else test_interval
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("test interval must be a positive integer")
    return {
        "fixed_learning_rate": fixed_learning_rate,
        "evaluation_interval_epochs": interval,
        "evaluate_epoch1": test_interval is None,
        "staged_schedule": fixed_learning_rate is None,
        "backbone_frozen": True,
    }


def _baseline_epoch_stage(optimizer, settings, epoch, protocol):
    if protocol["staged_schedule"]:
        return staged_epoch(optimizer, settings, epoch)
    lr = protocol["fixed_learning_rate"]
    for group in optimizer.param_groups:
        group["lr"] = lr
    return {"stage": "fixed_low_lr", "hard_balance_weight": 0.0,
            "lr_electronic": lr, "lr_saliency_head": lr}


def _baseline_should_evaluate(epoch, final_epoch, protocol):
    return ((epoch == 1 and protocol["evaluate_epoch1"])
            or epoch % protocol["evaluation_interval_epochs"] == 0 or epoch == final_epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/moe_staged_alpha_free.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-learning-rate", type=float, default=None)
    parser.add_argument("--test-interval", type=int, default=None,
                        help="Explicit historical interval/final policy; default keeps epoch1/config interval/final")
    args = parser.parse_args()
    s = load_settings(args.config)
    protocol = _resolve_baseline_protocol(s, args.fixed_learning_rate, args.test_interval)
    s.output_dir = args.run_dir.resolve()
    if s.output_dir.exists() and any(s.output_dir.iterdir()):
        raise FileExistsError("Use an empty run directory")
    s.output_dir.mkdir(parents=True)
    _seed(args.seed)
    s.random_seed = args.seed
    save_resolved_config(s)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    bundle = prepare_salicon(s, persist=True)
    loaded = load_vision_backbone(s, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    s.resolve_architecture(loaded.model)
    loaded.model.requires_grad_(False).eval()
    head = AlignedReadout(s.vision_hidden_size).to(loaded.device)
    model = FrozenQwenVisionTeacher(loaded, head)
    train_loader, test_loader = legacy.build_loaders(bundle, s, training=True)
    # The backbone has already initialized CUDA. Fork workers must not inherit
    # that context (and must not leave phantom GPU allocations after this run).
    use_spawn_workers(train_loader)
    use_spawn_workers(test_loader)
    adapter_lr = s.student_learning_rate if args.fixed_learning_rate is None else args.fixed_learning_rate
    decoder_lr = s.dense_head_learning_rate if args.fixed_learning_rate is None else args.fixed_learning_rate
    optim = torch.optim.AdamW([
        {"params": [*head.input_adapter.parameters(), *head.input_norm.parameters()], "name": "electronic", "lr": adapter_lr},
        {"params": list(head.decoder.parameters()), "name": "saliency_head", "lr": decoder_lr},
    ], weight_decay=s.weight_decay)
    manifest = {"git_commit": commit, "command": [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
                "architecture": "frozen_qwen24_adapter192_identical_progressive_decoder_v1",
                "parameter_audit": head.parameter_audit(), "seed": args.seed,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "qwen_frozen": True, "training_scope": "only adapter and decoder",
                "head_initialization": "random; no previous trained head loaded",
                "epochs_budget": s.student_epochs,
                "dataset_counts": {"train": len(bundle.train_records), "test": len(bundle.validation_records)},
                "selection_biased": True,
                "selection": "public test CC at " + ("epoch1/" if protocol["evaluate_epoch1"] else "")
                             + f"every{protocol['evaluation_interval_epochs']}/final",
                "training_protocol": protocol,
                "note": "Both systems have the same 197184-parameter adapter and 85412-parameter decoder; adapter appears before the optical body but after the frozen Qwen body. This is not an identical full-network/train-history ablation."}
    _write_json(s.output_dir / "run_manifest.json", manifest)
    history, best = [], -float("inf")
    try:
        for epoch in range(1, s.student_epochs+1):
            stage = _baseline_epoch_stage(optim, s, epoch, protocol)
            metrics = legacy._train_epoch("teacher", model, train_loader, loaded, s, optim)
            test = None
            if _baseline_should_evaluate(epoch, s.student_epochs, protocol):
                test, _ = legacy.evaluate_model(model, test_loader, loaded, s)
            payload = {"architecture": manifest["architecture"], "epoch": epoch, "head": head.state_dict(),
                       "train_metrics": metrics, "test_metrics": test, "manifest": manifest}
            torch.save(payload, s.output_dir / "last_checkpoint.pt")
            if test is not None and test["cc"] > best:
                best = test["cc"]
                torch.save(payload, s.output_dir / "best_checkpoint.pt")
            history.append({"epoch": epoch, **stage, **{f"train_{k}": v for k,v in metrics.items()},
                            **({f"test_{k}": v for k,v in test.items()} if test else {})})
            _write_csv(s.output_dir / "metrics/training_history.csv", history)
            print(f"[aligned Qwen] epoch={epoch}/{s.student_epochs} best_CC={best:.6f}", flush=True)
        selected = torch.load(s.output_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(selected["head"], strict=True)
        result, _ = legacy.evaluate_model(model, test_loader, loaded, s)
        _write_json(s.output_dir / "selected_checkpoint_test_evaluation.json", {
            **manifest, "metrics": result, "selected_epoch": selected["epoch"],
            "checkpoint_sha256": sha256_file(s.output_dir / "best_checkpoint.pt")})
    finally:
        model.close()


if __name__ == "__main__":
    main()
