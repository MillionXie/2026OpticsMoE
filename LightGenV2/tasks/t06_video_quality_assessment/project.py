"""Paths and compatibility contract for the current T06 implementation."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import yaml


TASK_DIR = Path(__file__).resolve().parent
LIGHTGEN_ROOT = TASK_DIR.parents[1]
REPO_ROOT = TASK_DIR.parents[2]
CURRENT_PROFILE = "temporal36_balanced"

_LOCAL_PATH_BINDINGS = {
    "dataset_root": ("data", "dataset_root", "dataset_root"),
    "manifest": ("data", "manifest", "manifest_path"),
    "vision_cache": ("data", "vision_cache", "vision_cache_path"),
    "language_cache": ("data", "language_cache", "language_cache_path"),
    "training_soft_targets": (
        "data",
        "training_soft_targets",
        "training_soft_targets_path",
    ),
    "initialization_checkpoint": (
        "training",
        "initialization_checkpoint",
        "initialization_checkpoint",
    ),
    "qwen_model_path": ("initialization", "qwen_model_path", "qwen_model_path"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_path(name: str) -> Path:
    path = TASK_DIR / "configs" / "lightgen" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown T06 profile: {path}")
    return path


def load_profile(name: str = CURRENT_PROFILE) -> dict[str, Any]:
    path = profile_path(name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Profile root must be a mapping: {path}")
    return raw


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_local_overrides() -> tuple[Path, dict[str, Any]]:
    path = LIGHTGEN_ROOT / "paths.local.yaml"
    if not path.is_file():
        return path, {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    task = raw.get("t06", {}) if isinstance(raw, dict) else {}
    if not isinstance(task, dict):
        raise ValueError(f"paths.local.yaml t06 must be a mapping: {path}")
    unknown = sorted(set(task) - set(_LOCAL_PATH_BINDINGS))
    if unknown:
        raise ValueError(f"Unknown paths.local.yaml t06 keys: {unknown}")
    return path, {key: value for key, value in task.items() if value not in (None, "")}


def verify_backend(profile: dict[str, Any]) -> dict[str, Any]:
    backend = profile["backend"]
    config = repo_path(backend["config"])
    package_root = REPO_ROOT / Path(str(backend["package"]).replace(".", "/"))
    files = {"config": (config, backend["config_sha256"])}
    files.update(
        {
            name: (package_root / name, expected)
            for name, expected in backend.get("source_sha256", {}).items()
        }
    )
    report = {}
    for name, (path, expected) in files.items():
        actual = sha256(path) if path.is_file() else None
        report[name] = {
            "path": str(path),
            "present": path.is_file(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
    return report


def inspect_profile(name: str = CURRENT_PROFILE) -> dict[str, Any]:
    profile = load_profile(name)
    backend = profile["backend"]
    config = repo_path(backend["config"])
    checkpoint = repo_path(profile["artifacts"]["canonical_checkpoint"])
    local_path, local_overrides = load_local_overrides()
    report: dict[str, Any] = {
        "profile": name,
        "profile_file": str(profile_path(name)),
        "backend_package": backend["package"],
        "backend_config": str(config),
        "backend_config_present": config.is_file(),
        "canonical_checkpoint": str(checkpoint),
        "canonical_checkpoint_present": checkpoint.is_file(),
        "canonical_checkpoint_expected_sha256": profile["artifacts"][
            "canonical_checkpoint_sha256"
        ],
        "paths_local_file": str(local_path),
        "paths_local_present": local_path.is_file(),
        "paths_local_active_overrides": local_overrides,
        "backend_integrity": verify_backend(profile),
    }
    if config.is_file():
        settings_module = importlib.import_module(f"{backend['package']}.settings")
        settings = settings_module.load_settings(config)
        resolved_values = {
            "dataset_root": settings.dataset_root,
            "manifest": settings.manifest_path,
            "vision_cache": settings.vision_cache_path,
            "language_cache": settings.language_cache_path,
            "training_soft_targets": settings.training_soft_targets_path,
            "initialization_checkpoint": settings.initialization_checkpoint,
        }
        for local_key, value in local_overrides.items():
            if local_key in resolved_values:
                resolved_values[local_key] = repo_path(value)
        report["resolved_inputs"] = {
            key: {
                "path": None if value is None else str(value),
                "present": None if value is None else Path(value).is_file() or Path(value).is_dir(),
            }
            for key, value in resolved_values.items()
        }
    if checkpoint.is_file():
        report["canonical_checkpoint_actual_sha256"] = sha256(checkpoint)
        report["canonical_checkpoint_hash_matches"] = (
            report["canonical_checkpoint_actual_sha256"]
            == report["canonical_checkpoint_expected_sha256"]
        )
    return report


_PATH_BINDINGS = {
    ("data", "dataset_root"): "dataset_root",
    ("data", "manifest"): "manifest_path",
    ("data", "vision_cache"): "vision_cache_path",
    ("data", "language_cache"): "language_cache_path",
    ("data", "quality_feature_cache"): "quality_feature_cache_path",
    ("data", "raw_frame_cache"): "raw_frame_cache_path",
    ("data", "vgg_feature_cache"): "vgg_feature_cache_path",
    ("data", "training_soft_targets"): "training_soft_targets_path",
    ("training", "initialization_checkpoint"): "initialization_checkpoint",
    ("initialization", "frame_stem_checkpoint"): "frame_stem_checkpoint",
    ("initialization", "qwen_model_path"): "qwen_model_path",
}


def materialize_launch_config(
    name: str,
    run_dir: Path,
    *,
    initialization_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], Path, str]:
    """Freeze inherited paths before moving output ownership to LightGenV2."""

    profile = load_profile(name)
    backend = profile["backend"]
    integrity = verify_backend(profile)
    mismatches = [key for key, value in integrity.items() if not value["matches"]]
    if mismatches:
        raise RuntimeError(
            "T06 verified backend changed or is incomplete: " + ", ".join(mismatches)
        )
    package = str(backend["package"])
    source_config = repo_path(backend["config"])
    settings_module = importlib.import_module(f"{package}.settings")
    settings = settings_module.load_settings(source_config)
    raw = settings_module._read_layered(source_config)  # same verified backend
    raw["output_dir"] = str(run_dir.resolve())
    for (section, key), attribute in _PATH_BINDINGS.items():
        raw.setdefault(section, {})[key] = (
            None if getattr(settings, attribute) is None else str(getattr(settings, attribute))
        )
    _, local_overrides = load_local_overrides()
    for local_key, value in local_overrides.items():
        section, key, _ = _LOCAL_PATH_BINDINGS[local_key]
        raw.setdefault(section, {})[key] = str(repo_path(value))
    if initialization_checkpoint is not None:
        raw.setdefault("training", {})["initialization_checkpoint"] = str(
            initialization_checkpoint.resolve()
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    destination = run_dir / "resolved_launch.yaml"
    destination.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    identity = {
        "profile": name,
        "profile_file": str(profile_path(name)),
        "backend_package": package,
        "backend_source_config": str(source_config),
        "backend_source_config_sha256": sha256(source_config),
        "resolved_launch_config": str(destination),
    }
    (run_dir / "launch_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return profile, destination, package


__all__ = [
    "CURRENT_PROFILE",
    "LIGHTGEN_ROOT",
    "REPO_ROOT",
    "TASK_DIR",
    "inspect_profile",
    "load_profile",
    "load_local_overrides",
    "materialize_launch_config",
    "profile_path",
    "repo_path",
    "sha256",
    "verify_backend",
]
