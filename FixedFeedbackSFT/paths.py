"""Stable repository paths for the reorganized fixed-feedback project.

The project is frequently executed from detached Git worktrees on the server,
so path discovery must not depend on a fixed number of parent directories.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_NAMES = (
    "d2nn_cifar100c10_fixed_feedback_20stage400",
    "d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400",
    "d2nn_cifar10_high_performance_optical_backbone",
    "qwen3_vl_patch_stem_8stage_optical_imagenet_backbone",
    "qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone",
    "qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone",
    "qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone",
    "qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa",
    "qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone",
)


def find_repository_root(start: str | Path | None = None) -> Path:
    """Find the checkout root in a normal clone or a linked Git worktree."""

    current = Path(start or __file__).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "FixedFeedbackSFT").is_dir()
            and (candidate / "experiments").is_dir()
            and (candidate / ".git").exists()
        ):
            return candidate
    raise RuntimeError(f"Could not locate the 2026OpticsMoE repository above {current}")


REPOSITORY_ROOT = find_repository_root()
PROJECTS_ROOT = REPOSITORY_ROOT / "FixedFeedbackSFT" / "projects"
REPORTS_ROOT = REPOSITORY_ROOT / "FixedFeedbackSFT" / "reports"
RUNS_ROOT = Path(
    os.environ.get(
        "FIXED_FEEDBACK_RUNS_ROOT",
        str(REPOSITORY_ROOT / "FixedFeedbackSFT" / "runs"),
    )
).expanduser().resolve()


def project_directory(name: str) -> Path:
    """Return a canonical physical project directory and reject typos early."""

    path = PROJECTS_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(f"Unknown fixed-feedback project: {path}")
    return path


def run_directory(project: str, run_name: str | None = None) -> Path:
    """Return the central, Git-ignored runtime path for a project."""

    if project not in PROJECT_NAMES:
        raise KeyError(f"Unknown fixed-feedback project: {project}")
    path = RUNS_ROOT / project
    return path if run_name is None else path / run_name


def resolve_repository_path(
    value: str | Path,
    *,
    base: str | Path | None = None,
) -> Path:
    """Resolve a config path and honor the optional central runs override."""

    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded.resolve()
    relative_runs = Path("FixedFeedbackSFT") / "runs"
    try:
        run_relative = expanded.relative_to(relative_runs)
    except ValueError:
        root = REPOSITORY_ROOT if base is None else Path(base)
        return (root / expanded).resolve()
    return (RUNS_ROOT / run_relative).resolve()


__all__ = [
    "PROJECTS_ROOT",
    "PROJECT_NAMES",
    "REPORTS_ROOT",
    "REPOSITORY_ROOT",
    "RUNS_ROOT",
    "find_repository_root",
    "project_directory",
    "resolve_repository_path",
    "run_directory",
]
