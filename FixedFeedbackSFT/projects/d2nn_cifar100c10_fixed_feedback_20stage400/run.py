from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .analysis import compare_methods
from .datasets import load_datasets, prepare_data
from .model import OpticalClassifier
from .settings import OpticalConfig, load_settings
from .training import evaluate_no_finetuning, finetune, pretrain


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoke_test() -> None:
    config = OpticalConfig(
        canvas_size=32,
        num_stages=3,
        wavelength_m=532e-9,
        pixel_size_m=16e-6,
        propagation_distance_m=0.05,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        layernorm_eps=1e-5,
        residual_main_init=0.9,
        residual_skip_init=0.1,
        readout_pool_size=8,
        readout_hidden_dim=16,
        num_output_classes=10,
    )
    device = _device()
    model = OpticalClassifier(config).to(device)
    images = torch.rand(2, 1, 32, 32, device=device)
    labels = torch.tensor([1, 4], device=device)
    pretrained = model.snapshot_feedback_phases()
    gradients: dict[str, list[torch.Tensor]] = {}
    outputs: dict[str, torch.Tensor] = {}
    for mode in ("bp", "fa_pretrained", "fa_random"):
        model.configure_feedback(
            mode,
            pretrained_phases=pretrained if mode == "fa_pretrained" else None,
            random_seed=1234,
        )
        model.zero_grad(set_to_none=True)
        logits = model(images)
        F.cross_entropy(logits, labels).backward()
        outputs[mode] = logits.detach()
        gradients[mode] = [stage.raw_phase.grad.detach().clone() for stage in model.stages]
    if not torch.equal(outputs["bp"], outputs["fa_pretrained"]):
        raise AssertionError("Feedback mode changed the forward output")
    for bp, fixed in zip(gradients["bp"], gradients["fa_pretrained"], strict=True):
        if not torch.allclose(bp, fixed, rtol=2e-4, atol=2e-6):
            raise AssertionError("FA-pretrained does not match BP at the initialization")
    print(
        json.dumps(
            {
                "smoke": "passed",
                "device": str(device),
                "output_shape": list(outputs["bp"].shape),
                "phase_parameters": model.parameter_report()["phase"],
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
        choices=("prepare_data", "pretrain", "finetune", "no_finetune", "run", "compare", "all", "smoke"),
    )
    parser.add_argument("--method", choices=("bp", "fa_pretrained", "fa_random", "no_finetune"))
    parser.add_argument("--seed", type=int, help="Run only one configured fine-tuning seed")
    parser.add_argument("--force", action="store_true", help="Overwrite completed training for the requested phase")
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
    bundle = load_datasets(settings, prepare=True)
    print(
        f"pretrain={len(bundle.pretrain_train)}/{len(bundle.pretrain_validation)} "
        f"downstream={len(bundle.finetune_train)}/{len(bundle.finetune_validation)}/{len(bundle.finetune_test)}",
        flush=True,
    )
    if args.phase == "pretrain":
        pretrain(settings, bundle, device, force=args.force)
        return 0
    if args.phase == "no_finetune":
        evaluate_no_finetuning(settings, bundle, device)
        return 0
    if args.phase == "finetune":
        if args.method not in {"bp", "fa_pretrained", "fa_random"}:
            raise SystemExit("--phase finetune requires --method bp/fa_pretrained/fa_random")
        seeds = (args.seed,) if args.seed is not None else settings.training.finetune_seeds
        for seed in seeds:
            finetune(settings, bundle, device, method=args.method, seed=seed, force=args.force)
        return 0
    if args.phase == "run":
        if args.method is None:
            raise SystemExit("--phase run requires --method")
        pretrain(settings, bundle, device, force=False)
        if args.method == "no_finetune":
            evaluate_no_finetuning(settings, bundle, device)
        else:
            seeds = (args.seed,) if args.seed is not None else settings.training.finetune_seeds
            for seed in seeds:
                finetune(settings, bundle, device, method=args.method, seed=seed, force=args.force)
        return 0
    if args.phase == "all":
        pretrain(settings, bundle, device, force=False)
        evaluate_no_finetuning(settings, bundle, device)
        for method in ("bp", "fa_pretrained", "fa_random"):
            for seed in settings.training.finetune_seeds:
                finetune(settings, bundle, device, method=method, seed=seed, force=args.force)
        compare_methods(settings)
        return 0
    raise AssertionError(args.phase)
