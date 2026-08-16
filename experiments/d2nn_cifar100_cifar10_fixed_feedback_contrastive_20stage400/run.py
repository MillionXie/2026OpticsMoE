from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analysis import compare_methods
from .datasets import load_datasets, prepare_data
from .losses import contrastive_transfer_loss
from .model import OpticalEmbeddingNetwork
from .settings import OpticalConfig, load_settings
from .training import (
    build_model,
    build_optimizer,
    evaluate_no_finetuning,
    finetune,
    pretrain,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoke_test() -> None:
    config = OpticalConfig(
        canvas_size=24,
        num_stages=3,
        wavelength_m=532e-9,
        pixel_size_m=16e-6,
        propagation_distance_m=0.05,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        layernorm_eps=1e-5,
        residual_main_init=0.35,
        residual_skip_init=0.65,
        readout_pool_size=4,
        embedding_dim=12,
        embedding_dropout=0.1,
    )
    device = _device()
    model = OpticalEmbeddingNetwork(config).to(device)
    images = torch.rand(8, 1, 24, 24, device=device)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    model.train()
    features = model(images).reshape(4, 2, -1)
    loss, parts = contrastive_transfer_loss(
        features,
        labels,
        contrastive_temperature=0.1,
        prototype_temperature=0.1,
        supcon_weight=1.0,
        prototype_weight=0.5,
    )
    loss.backward()
    if not all(stage.raw_phase.grad is not None and torch.isfinite(stage.raw_phase.grad).all() for stage in model.stages):
        raise AssertionError("Optical phase gradient is missing or non-finite")
    print(
        json.dumps(
            {
                "smoke": "passed",
                "device": str(device),
                "embedding_shape": list(features.shape),
                "embedding_norm_mean": float(features.norm(dim=-1).mean()),
                "loss": float(loss),
                "batch_prototype_accuracy": float(parts["batch_prototype_accuracy"]),
                "residual_weights": model.residual_weights().detach().cpu().tolist(),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("prepare_data", "pretrain", "finetune", "no_finetune", "compare", "all", "smoke", "formal_smoke"),
    )
    parser.add_argument("--method", choices=("bp", "fa_pretrained", "fa_random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "smoke":
        smoke_test()
        return 0
    if args.config is None:
        raise SystemExit("--config is required except for --phase smoke")
    settings = load_settings(args.config)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "resolved_config.json").write_text(
        json.dumps(settings.resolved_dict(), indent=2), encoding="utf-8"
    )
    device = _device()
    print(f"device={device} output_dir={settings.output_dir}", flush=True)
    if args.phase == "prepare_data":
        print(json.dumps(prepare_data(settings), indent=2), flush=True)
        return 0
    if args.phase == "compare":
        compare_methods(settings)
        return 0
    if args.phase == "formal_smoke":
        # The formal model check is deliberately synthetic so it never waits
        # for a dataset download or mutates a training split.
        model = build_model(settings, device)
        optimizer = build_optimizer(model, settings, finetune=False)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        labels = torch.tensor([0, 0, 1, 1], device=device)
        images = torch.rand(8, 1, 32, 32, device=device)
        embeddings = model(images).reshape(4, 2, -1)
        loss, parts = contrastive_transfer_loss(
            embeddings,
            labels,
            contrastive_temperature=settings.loss.contrastive_temperature,
            prototype_temperature=settings.loss.prototype_temperature,
            supcon_weight=1.0,
            prototype_weight=0.5,
        )
        loss.backward()
        optimizer.step()
        print(
            json.dumps(
                {
                    "formal_one_batch": "passed",
                    "loss": float(loss.detach()),
                    "batch_prototype_accuracy": float(parts["batch_prototype_accuracy"]),
                    "phase_gradients_finite": all(
                        stage.raw_phase.grad is not None and torch.isfinite(stage.raw_phase.grad).all()
                        for stage in model.stages
                    ),
                    "embedding_shape": list(embeddings.shape),
                    "embedding_norm_mean": float(embeddings.norm(dim=-1).mean()),
                    "phase_parameters": model.parameter_report()["phase"],
                    "residual_weights_mean": model.residual_weights().detach().mean(dim=0).cpu().tolist(),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    bundle = load_datasets(settings, prepare=True)
    print(
        f"CIFAR-100 pretrain={len(bundle.pretrain_train)}/{len(bundle.pretrain_validation)}; "
        f"CIFAR-10 train/support/validation/test={len(bundle.finetune_train)}/"
        f"{len(bundle.prototype_support)}/{len(bundle.finetune_validation)}/{len(bundle.finetune_test)}",
        flush=True,
    )
    if args.phase == "pretrain":
        pretrain(settings, bundle, device, force=args.force)
    elif args.phase == "no_finetune":
        evaluate_no_finetuning(settings, bundle, device)
    elif args.phase == "finetune":
        if args.method is None:
            raise SystemExit("--phase finetune requires --method")
        seeds = (args.seed,) if args.seed is not None else settings.training.finetune_seeds
        for seed in seeds:
            finetune(settings, bundle, device, method=args.method, seed=seed, force=args.force)
    elif args.phase == "all":
        pretrain(settings, bundle, device, force=False)
        evaluate_no_finetuning(settings, bundle, device)
        for method in ("bp", "fa_pretrained", "fa_random"):
            for seed in settings.training.finetune_seeds:
                finetune(settings, bundle, device, method=method, seed=seed, force=args.force)
        compare_methods(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
