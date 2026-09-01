from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)

from .settings import EXPERIMENT_DIR, REPO_ROOT, load_settings


SOURCE_FORMAT = "p12-p11-random-body-source-v1"
SOURCE_REGIME = "random_p11_body_frozen_qwen_stem"
DEFAULT_BASE_CONFIG = EXPERIMENT_DIR / "configs" / "base_50e.yaml"
_P11_SIGNATURE = torch.tensor([11, 1, 2, 4], dtype=torch.int64)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes and bytes independent of ``torch.save``."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Backbone state must be a string-to-tensor mapping")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        # Flatten first so scalar gate parameters also have a last dimension
        # that can be reinterpreted as bytes.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _subset_digest(
    state: Mapping[str, torch.Tensor], *, stem: bool
) -> str:
    subset = {
        name: value
        for name, value in state.items()
        if name.startswith("stem.") == stem
    }
    if not subset:
        raise RuntimeError("Scratch source checkpoint has an empty state subset")
    return state_dict_sha256(subset)


def inspect_scratch_source(path: str | Path) -> dict[str, Any]:
    """Validate a generated source and return its immutable provenance."""

    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Scratch source checkpoint must contain a mapping")
    if payload.get("format") != SOURCE_FORMAT:
        raise RuntimeError(
            f"Unsupported scratch source format {payload.get('format')!r}; "
            f"expected {SOURCE_FORMAT!r}"
        )
    if payload.get("source_regime") != SOURCE_REGIME:
        raise RuntimeError("Scratch source checkpoint has the wrong source_regime")
    init_seed = payload.get("init_seed")
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or init_seed < 0:
        raise RuntimeError("Scratch source init_seed must be a non-negative integer")
    state = payload.get("backbone")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise TypeError("Scratch source backbone must be a string-to-tensor mapping")
    signature = state.get("p11_separable_architecture_signature")
    if not isinstance(signature, torch.Tensor) or not torch.equal(
        signature.detach().cpu(), _P11_SIGNATURE
    ):
        raise RuntimeError("Scratch source does not carry the P11 architecture signature")

    state_digest = state_dict_sha256(state)
    stem_digest = _subset_digest(state, stem=True)
    non_stem_digest = _subset_digest(state, stem=False)
    expected = {
        "backbone_state_sha256": state_digest,
        "stem_state_sha256": stem_digest,
        "non_stem_state_sha256": non_stem_digest,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Scratch source semantic digest mismatch: {mismatches}")
    stem_checkpoint_sha256 = str(payload.get("stem_checkpoint_sha256", ""))
    if len(stem_checkpoint_sha256) != 64:
        raise RuntimeError("Scratch source has an invalid frozen stem SHA-256")
    report = payload.get("model_report")
    if not isinstance(report, Mapping) or report.get("optical_mixer_variant") != (
        "separable_token_channel_axis"
    ):
        raise RuntimeError("Scratch source has an incompatible P11 model report")
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get(
        "copied_from_imagenet_pretraining"
    ) is not False:
        raise RuntimeError("Scratch source does not attest fresh body initialization")
    return {
        "path": str(resolved),
        "checkpoint_sha256": sha256_file(resolved),
        "source_regime": SOURCE_REGIME,
        "init_seed": init_seed,
        "stem_checkpoint_sha256": stem_checkpoint_sha256,
        **expected,
    }


def export_random_p11_source(
    *,
    stem_checkpoint: str | Path,
    output: str | Path,
    p11_config: Mapping[str, Any],
    init_seed: int = 2026,
) -> dict[str, Any]:
    """Create a fresh P11 body without reading any ImageNet backbone checkpoint.

    The Qwen patch/position stem is loaded from the same frozen artifact used by
    the pretrained control. Every adapter, phase plane, mixer, residual gate and
    ImageNet readout is constructed afresh under ``init_seed``; the readout is
    then omitted through the normal ``backbone_state_dict`` contract.
    """

    if isinstance(init_seed, bool) or int(init_seed) < 0:
        raise ValueError("init_seed must be a non-negative integer")
    init_seed = int(init_seed)
    stem_path = Path(stem_checkpoint).expanduser().resolve()
    if not stem_path.is_file():
        raise FileNotFoundError(f"Frozen Qwen stem checkpoint does not exist: {stem_path}")
    config = dict(p11_config)
    config["seed"] = init_seed

    # Isolate the export from caller RNG state. The P11 constructor also uses
    # per-plane CPU generators for optical phases; the fork covers adapter and
    # electronic mixer initializers that use PyTorch's global CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = QwenStemSeparableOpticalImageNetBackbone(stem_path, config)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.backbone_state_dict().items()
    }
    state_digest = state_dict_sha256(state)
    stem_digest = _subset_digest(state, stem=True)
    non_stem_digest = _subset_digest(state, stem=False)
    payload: dict[str, Any] = {
        "format": SOURCE_FORMAT,
        "source_regime": SOURCE_REGIME,
        "init_seed": init_seed,
        "backbone": state,
        "backbone_state_sha256": state_digest,
        "stem_state_sha256": stem_digest,
        "non_stem_state_sha256": non_stem_digest,
        "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
        "model_report": model.parameter_report(),
        "architecture_signature": [11, 1, 2, 4],
        "initialization": {
            "copied_from_imagenet_pretraining": False,
            "frozen_qwen_stem_from_checkpoint": True,
            "fresh_components": [
                "adapter",
                "eight_optical_phase_planes",
                "eight_slim_spatial_token_mixers",
                "eight_constrained_optical_gates",
            ],
        },
        "feature_contract": {
            "input": "CLIP-normalized RGB [B,3,224,224]",
            "final": "three latent optical banks [B,3,224,224]",
            "stages": "tuple of eight [B,3,224,224] OEO feature maps",
            "qwen_transformer_required": False,
        },
    }

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = inspect_scratch_source(output_path)
        requested = {
            "source_regime": SOURCE_REGIME,
            "init_seed": init_seed,
            "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
            "backbone_state_sha256": state_digest,
            "stem_state_sha256": stem_digest,
            "non_stem_state_sha256": non_stem_digest,
        }
        mismatch = {
            key: (existing.get(key), value)
            for key, value in requested.items()
            if existing.get(key) != value
        }
        if mismatch:
            raise FileExistsError(
                f"Refusing to replace a different scratch source at {output_path}: {mismatch}"
            )
        return {**existing, "status": "reused"}

    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**inspect_scratch_source(output_path), "status": "created"}


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def render_scratch_config(
    *,
    base_config: str | Path,
    source_checkpoint: str | Path,
    output: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Render a runnable config only after the real source SHA is known."""

    base_path = Path(base_config).expanduser().resolve()
    base_settings = load_settings(base_path)
    source = inspect_scratch_source(source_checkpoint)
    stem_path = base_settings.paths.stem_checkpoint
    if not stem_path.is_file():
        raise FileNotFoundError(f"Frozen Qwen stem checkpoint does not exist: {stem_path}")
    actual_stem_sha = sha256_file(stem_path)
    if actual_stem_sha != source["stem_checkpoint_sha256"]:
        raise RuntimeError(
            "Scratch source was built from a different frozen Qwen stem: "
            f"source={source['stem_checkpoint_sha256']}, config={actual_stem_sha}"
        )

    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("paths"), dict):
        raise ValueError("Base P12 config must contain a paths mapping")
    source_path = Path(source_checkpoint).expanduser().resolve()
    root_path = Path(output_root).expanduser()
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path
    raw["paths"]["source_backbone"] = _portable_path(source_path)
    raw["paths"]["source_backbone_sha256"] = source["checkpoint_sha256"]
    raw["paths"]["output_root"] = _portable_path(root_path)
    raw["scratch_control"] = {
        "source_regime": source["source_regime"],
        "init_seed": source["init_seed"],
        "no_imagenet_backbone_pretraining": True,
        "frozen_qwen_stem_checkpoint_sha256": source["stem_checkpoint_sha256"],
        "backbone_state_sha256": source["backbone_state_sha256"],
        "non_stem_state_sha256": source["non_stem_state_sha256"],
        "fa_pretrained_method_label_in_this_control": "fa_source_init",
    }

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    resolved = load_settings(output_path)
    if resolved.paths.source_backbone != source_path:
        raise RuntimeError("Rendered config did not resolve to the generated scratch source")
    if resolved.paths.source_backbone_sha256 != source["checkpoint_sha256"]:
        raise RuntimeError("Rendered config did not lock the generated checkpoint SHA-256")
    return {
        "status": "rendered",
        "config": str(output_path),
        "output_root": str(resolved.paths.output_root),
        **source,
    }


def _base_p11_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    settings = load_settings(path)
    return settings.paths.stem_checkpoint, settings.p11_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and lock a fresh P11-body source for the P12 scratch control."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export the deterministic random body")
    export.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    export.add_argument("--stem-checkpoint", type=Path)
    export.add_argument("--init-seed", type=int, default=2026)
    export.add_argument("--output", type=Path, required=True)

    render = subparsers.add_parser(
        "render-config", help="Render a config with the generated source's real SHA-256"
    )
    render.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    render.add_argument("--source-checkpoint", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        default_stem, p11_config = _base_p11_config(args.base_config)
        report = export_random_p11_source(
            stem_checkpoint=(
                default_stem if args.stem_checkpoint is None else args.stem_checkpoint
            ),
            output=args.output,
            p11_config=p11_config,
            init_seed=args.init_seed,
        )
    else:
        report = render_scratch_config(
            base_config=args.base_config,
            source_checkpoint=args.source_checkpoint,
            output=args.output,
            output_root=args.output_root,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_FORMAT",
    "SOURCE_REGIME",
    "export_random_p11_source",
    "inspect_scratch_source",
    "render_scratch_config",
    "sha256_file",
    "state_dict_sha256",
]
