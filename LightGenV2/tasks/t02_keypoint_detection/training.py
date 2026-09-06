"""Bind the audited LightGen T02 graph to the verified LSP trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router import training as _base

from .modeling import architecture_label, build_student, initialize_student, optimizer


def _bind() -> None:
    # Each formal run is a separate Python process.  Runtime binding lets us
    # retain the already-tested LSP metric/loss loop while the task owns the
    # graph, initialization and optimizer contracts.
    _base.architecture_label = architecture_label
    _base.build_router_student = build_student
    _base.load_common_initialization = initialize_student
    _base._optimizer = optimizer


def train(loaded: Any, bundle: Any, settings: Any) -> dict[str, Any]:
    _bind()
    return _base.train(loaded, bundle, settings)


def evaluate_selected_checkpoint(
    loaded: Any, bundle: Any, settings: Any, checkpoint: Path
) -> dict[str, Any]:
    _bind()
    return _base.evaluate_selected_checkpoint(loaded, bundle, settings, checkpoint)


__all__ = ["evaluate_selected_checkpoint", "train"]
