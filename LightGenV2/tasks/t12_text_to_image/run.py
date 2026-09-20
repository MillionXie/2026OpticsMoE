"""T12 command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .feature_cache import build_feature_cache
from .losses import conditional_vae_loss
from .modeling import build_model
from .settings import TASK_DIR, Settings, load_settings
from .training import seed_everything, train


PROFILES = {
    "lightgen": "lightgen_parallel.yaml",
    "lightgen_gan": "lightgen_parallel_gan.yaml",
    "baseline": "qwen_vae_baseline.yaml",
    "baseline_gan": "qwen_vae_baseline_gan.yaml",
    "smoke": "smoke.yaml",
}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=TASK_DIR, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def smoke(settings: Settings) -> dict[str, Any]:
    """Dependency-light forward/backward contract for both comparison rows."""

    seed_everything(settings.seed)
    reports = {}
    for variant in ("lightgen_parallel", "qwen_vae_baseline"):
        current = dataclasses.replace(
            settings,
            variant=variant,
            optical_backend="compact_fft" if variant == "lightgen_parallel" else "none",
        )
        current.validate()
        model = build_model(current)
        text = torch.randn(2, current.text_dim)
        target = torch.randn(2, current.latent_channels, current.latent_size, current.latent_size)
        output = model.forward_train(text, target)
        loss, metrics = conditional_vae_loss(output, target, kl_weight=current.kl_weight, free_bits=0)
        loss.backward()
        reports[variant] = {
            "output_shape": list(output.predicted_latent.shape),
            "loss": metrics,
            "architecture": model.architecture_report(),
            "gradient_parameters": sum(parameter.grad is not None for parameter in model.parameters()),
        }
    return {"status": "passed", "variants": reports}


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings(TASK_DIR / "configs" / PROFILES[args.profile])
    if args.data_dir:
        settings.data_dir = Path(args.data_dir).expanduser().resolve()
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    if args.qwen_checkpoint:
        settings.qwen_checkpoint = Path(args.qwen_checkpoint).expanduser().resolve()
    if args.vae_checkpoint:
        settings.vae_checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
    _write_json(settings.output_dir / "run_manifest.json", {
        "schema_version": 1,
        "task": "t12_text_to_image",
        "profile": args.profile,
        "phase": args.phase,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "command": [sys.executable, *sys.argv],
        "device": str(device),
    })
    if args.phase == "smoke":
        result = smoke(settings)
    elif args.phase == "cache":
        result = build_feature_cache(settings, device, force=args.force)
    elif args.phase == "train":
        result = train(settings, device)
    elif args.phase == "evaluate":
        from .evaluation import evaluate
        checkpoint = (
            Path(args.checkpoint).expanduser().resolve()
            if args.checkpoint else settings.output_dir / "best_checkpoint.pt"
        )
        result = evaluate(settings, checkpoint, settings.output_dir / "evaluation", device)
    else:
        build_feature_cache(settings, device, force=args.force)
        result = train(settings, device)
    _write_json(settings.output_dir / f"{args.phase}_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGenV2 T12 single-pass text-to-image")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=("smoke", "cache", "train", "evaluate", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--qwen-checkpoint", default=None)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
