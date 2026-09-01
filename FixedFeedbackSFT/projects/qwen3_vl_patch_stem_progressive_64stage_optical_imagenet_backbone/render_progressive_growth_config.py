from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from FixedFeedbackSFT.paths import REPOSITORY_ROOT

from .migration import P13_TRAINING_CHECKPOINT_FORMAT


TRANSITIONS = {32: 16, 64: 32, 100: 64}
TRAINING_GEOMETRY = {
    32: {"batch_size": 12, "validation_batch_size": 24, "gradient_accumulation_steps": 4},
    64: {"batch_size": 6, "validation_batch_size": 12, "gradient_accumulation_steps": 8},
    100: {"batch_size": 4, "validation_batch_size": 8, "gradient_accumulation_steps": 12},
}
EXPECTED_WORLD_SIZE = 4
EXPECTED_GLOBAL_BATCH = 192


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} must be a non-empty string")
    return value


def inspect_parent_checkpoint(
    checkpoint: str | Path,
    *,
    target_depth: int,
) -> dict[str, Any]:
    """Reject any source other than the preceding formal best-full-depth model."""

    target = int(target_depth)
    if target not in TRANSITIONS:
        raise ValueError(f"target_depth must be one of {tuple(TRANSITIONS)}")
    expected_source_depth = TRANSITIONS[target]
    source = Path(checkpoint).expanduser().resolve()
    if source.name != "best_full_depth.pt":
        raise RuntimeError("Progressive growth source must be named best_full_depth.pt")
    if not source.is_file():
        raise FileNotFoundError(
            f"Previous depth {expected_source_depth} has no best_full_depth.pt: {source}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload = _mapping(payload, name="parent checkpoint")
    if payload.get("format") != P13_TRAINING_CHECKPOINT_FORMAT:
        raise RuntimeError("Parent is not a formal P13 ImageNet training checkpoint")
    if payload.get("checkpoint_role") != "best_full_depth":
        raise RuntimeError("Parent checkpoint role must be best_full_depth")

    model_config = _mapping(payload.get("model_config"), name="parent model_config")
    if model_config.get("num_stages") != expected_source_depth:
        raise RuntimeError(
            f"Target {target} requires source depth {expected_source_depth}"
        )
    state = _mapping(payload.get("model"), name="parent complete model state")
    signature = state.get("p13_progressive_architecture_signature")
    if not isinstance(signature, torch.Tensor) or tuple(signature.shape) != (4,):
        raise RuntimeError("Parent model has no valid P13 architecture signature")
    signature_values = tuple(int(value) for value in signature.tolist())
    if signature_values != (13, 1, 2, expected_source_depth):
        raise RuntimeError("Parent architecture signature has another depth")

    depth_alpha = _mapping(payload.get("depth_alpha"), name="parent depth_alpha")
    if depth_alpha.get("all_full_depth") is not True:
        raise RuntimeError("Parent is not at alpha-one full depth")
    report = _mapping(payload.get("model_report"), name="parent model_report")
    if report.get("architecture") != "p13_progressive_p11_token_channel":
        raise RuntimeError("Parent model_report has another architecture")
    if report.get("num_stages") != expected_source_depth:
        raise RuntimeError("Parent model_report depth is inconsistent")
    if report.get("depth_alpha") != depth_alpha:
        raise RuntimeError("Parent model_report alpha metadata is inconsistent")

    migration = _mapping(
        payload.get("migration_manifest"), name="parent migration_manifest"
    )
    expected_grandparent = 8 if expected_source_depth == 16 else expected_source_depth // 2
    if migration.get("target_num_stages") != expected_source_depth:
        raise RuntimeError("Parent migration target depth is inconsistent")
    if migration.get("source_depth") != expected_grandparent:
        raise RuntimeError("Parent was not grown from the immediately preceding depth")
    if migration.get("target_architecture_signature") != [
        13,
        1,
        2,
        expected_source_depth,
    ]:
        raise RuntimeError("Parent migration target signature is inconsistent")
    if report.get("migration_manifest") != migration:
        raise RuntimeError("Parent model_report migration provenance is inconsistent")
    initialization = _mapping(
        payload.get("initialization_manifest"),
        name="parent initialization_manifest",
    )
    if initialization.get("source_depth") != expected_grandparent:
        raise RuntimeError("Parent initialization source depth is inconsistent")
    if initialization.get("target_depth") != expected_source_depth:
        raise RuntimeError("Parent initialization target depth is inconsistent")
    if initialization.get("migration") != migration:
        raise RuntimeError("Parent initialization/migration manifests disagree")

    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise RuntimeError("Parent checkpoint has no positive epoch")
    config_digest = _nonempty_string(
        payload.get("config_digest"), name="parent config_digest"
    )
    stem_sha256 = _nonempty_string(
        payload.get("stem_checkpoint_sha256"),
        name="parent stem_checkpoint_sha256",
    )
    feedback = _mapping(payload.get("feedback"), name="parent feedback")
    feedback_method = feedback.get("method")
    if feedback_method not in {"bp_current", "fa_source", "fa_random"}:
        raise RuntimeError("Parent feedback method is unsupported or missing")
    feedback_manifest = _mapping(
        feedback.get("manifest"), name="parent feedback manifest"
    )
    feedback_manifest_sha256 = canonical_json_sha256(feedback_manifest)
    if feedback.get("manifest_sha256") != feedback_manifest_sha256:
        raise RuntimeError("Parent feedback manifest SHA-256 is inconsistent")
    if feedback_manifest.get("depth") != expected_source_depth:
        raise RuntimeError("Parent feedback manifest depth is inconsistent")

    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "source_depth": expected_source_depth,
        "target_depth": target,
        "epoch": epoch,
        "config_digest": config_digest,
        "stem_checkpoint_sha256": stem_sha256,
        "feedback_method": feedback_method,
        "feedback_manifest_sha256": feedback_manifest_sha256,
    }


def _portable_path(path: Path, repository: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError:
        return str(resolved)


def render_config(
    *,
    template_config: str | Path,
    parent_checkpoint: str | Path,
    target_depth: int,
    output_dir: str | Path,
    repository: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository_path = Path(repository).expanduser().resolve()
    template_path = Path(template_config).expanduser().resolve()
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise RuntimeError("Progressive config template must contain a mapping")
    identity = inspect_parent_checkpoint(
        parent_checkpoint,
        target_depth=target_depth,
    )
    target = int(target_depth)
    geometry = TRAINING_GEOMETRY[target]
    config = dict(template)
    config["output_dir"] = _portable_path(Path(output_dir), repository_path)
    config["model"] = dict(template["model"])
    config["model"]["num_stages"] = target
    config["initialization"] = {
        "mode": "progressive_growth",
        "source_training_checkpoint": _portable_path(
            Path(identity["path"]), repository_path
        ),
        "expected_source_training_sha256": identity["sha256"],
        "expected_source_depth": identity["source_depth"],
        "expected_source_epoch": identity["epoch"],
        "expected_source_config_digest": identity["config_digest"],
        "expected_source_feedback_method": identity["feedback_method"],
        "expected_source_feedback_manifest_sha256": identity[
            "feedback_manifest_sha256"
        ],
    }
    config["progressive_growth_guard"] = {
        "format": "p13-progressive-config-guard-v1",
        "renderer_sha256": sha256_file(Path(__file__)),
        "template_config": _portable_path(template_path, repository_path),
        "template_sha256": sha256_file(template_path),
        "parent_checkpoint": config["initialization"][
            "source_training_checkpoint"
        ],
        "parent_checkpoint_sha256": identity["sha256"],
        "source_depth": identity["source_depth"],
        "target_depth": target,
        "parent_feedback_method": identity["feedback_method"],
        "parent_feedback_manifest_sha256": identity[
            "feedback_manifest_sha256"
        ],
    }
    config["training"] = dict(template["training"])
    config["training"].update(geometry)
    config["training"]["expected_world_size"] = EXPECTED_WORLD_SIZE
    config["training"]["expected_effective_global_batch"] = EXPECTED_GLOBAL_BATCH
    actual_global_batch = (
        config["training"]["batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * config["training"]["expected_world_size"]
    )
    if actual_global_batch != EXPECTED_GLOBAL_BATCH:
        raise RuntimeError("Rendered config does not preserve global batch 192")
    return config, identity


def write_or_verify_config(path: str | Path, config: Mapping[str, Any]) -> str:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        existing = yaml.safe_load(destination.read_text(encoding="utf-8"))
        if existing != dict(config):
            raise RuntimeError(
                "Existing generated config differs from the current guarded source; "
                "refusing overwrite"
            )
        return "verified_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "rendered_new"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render or verify one guarded P13 progressive-growth config only "
            "after the immediately preceding best_full_depth checkpoint exists"
        )
    )
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--target-depth", type=int, choices=tuple(TRANSITIONS), required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, identity = render_config(
        template_config=args.template_config,
        parent_checkpoint=args.parent_checkpoint,
        target_depth=args.target_depth,
        output_dir=args.output_dir,
        repository=args.repository,
    )
    action = write_or_verify_config(args.output_config, config)
    print(
        json.dumps(
            {
                "action": action,
                "output_config": str(args.output_config.expanduser().resolve()),
                "parent": identity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
