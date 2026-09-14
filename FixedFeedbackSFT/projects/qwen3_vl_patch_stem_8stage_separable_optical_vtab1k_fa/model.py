from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.model import (
    P11DownstreamModel,
)


class P11VtabModel(P11DownstreamModel):
    """Classification-only P11 transfer model for any VTAB label space."""

    def __init__(
        self,
        *,
        stem_checkpoint: str | Path,
        source_checkpoint: str | Path,
        p11_config: Mapping[str, Any],
        task_name: str,
        num_classes: int,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            stem_checkpoint=stem_checkpoint,
            source_checkpoint=source_checkpoint,
            p11_config=p11_config,
            task="caltech101",
            num_outputs=int(num_classes),
            global_hidden_dim=int(head_hidden_dim),
        )
        self.task_name = str(task_name)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report["task"] = self.task_name
        report["benchmark"] = "VTAB-1k"
        return report


__all__ = ["P11VtabModel"]
