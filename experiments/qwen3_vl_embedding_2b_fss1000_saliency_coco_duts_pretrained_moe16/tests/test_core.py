from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16.settings import (
    EXPERIMENT_DIR,
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_fss1000_saliency_coco_duts_pretrained_moe16.training import (
    load_duts_initialization,
)


def test_config_keeps_three_stage_source_architecture() -> None:
    settings = load_settings(
        EXPERIMENT_DIR / "configs" / "fss1000_finetune.yaml"
    )
    assert settings.expert_layers == 3
    assert settings.num_experts == 16
    assert settings.top_k == 4
    assert settings.image_size == 224
    assert settings.detector_output_size == 224
    assert settings.output_dir.parent == EXPERIMENT_DIR / "runs"


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.recombiner = nn.Linear(2, 2)

    def specification(self) -> dict[str, object]:
        return {
            "expert_stages": 3,
            "experts_per_stage": 16,
            "top_k": 4,
            "expert_size": 224,
            "active_size": 986,
            "canvas_size": 1026,
            "ccd_shape": [224, 224],
        }


def test_transfer_checkpoint_restores_backbone_recombiner_and_head(
    tmp_path: Path,
) -> None:
    source = SimpleNamespace(backbone=_FakeBackbone(), head=nn.Linear(2, 1))
    with torch.no_grad():
        source.backbone.core.weight.fill_(1.0)
        source.backbone.recombiner.weight.fill_(2.0)
        source.head.weight.fill_(3.0)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "checkpoint_type": "duts_saliency_pretraining",
            "epoch": 17,
            "backbone": {
                "core_state_dict": source.backbone.core.state_dict(),
                "recombiner_state_dict": (
                    source.backbone.recombiner.state_dict()
                ),
                "architecture": source.backbone.specification(),
            },
            "head_state_dict": source.head.state_dict(),
        },
        checkpoint,
    )

    target = SimpleNamespace(backbone=_FakeBackbone(), head=nn.Linear(2, 1))
    report = load_duts_initialization(target, checkpoint)
    torch.testing.assert_close(
        target.backbone.core.weight,
        source.backbone.core.weight,
    )
    torch.testing.assert_close(
        target.backbone.recombiner.weight,
        source.backbone.recombiner.weight,
    )
    torch.testing.assert_close(target.head.weight, source.head.weight)
    assert report["source_epoch"] == 17
    assert report["optimizer_restored"] is False


def test_architecture_mismatch_is_rejected(tmp_path: Path) -> None:
    target = SimpleNamespace(backbone=_FakeBackbone(), head=nn.Linear(2, 1))
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "checkpoint_type": "duts_saliency_pretraining",
            "epoch": 1,
            "backbone": {
                "core_state_dict": target.backbone.core.state_dict(),
                "recombiner_state_dict": (
                    target.backbone.recombiner.state_dict()
                ),
                "architecture": {
                    **target.backbone.specification(),
                    "expert_stages": 1,
                },
            },
            "head_state_dict": target.head.state_dict(),
        },
        checkpoint,
    )
    try:
        load_duts_initialization(target, checkpoint)
    except RuntimeError as exc:
        assert "expert_stages" in str(exc)
    else:
        raise AssertionError("Architecture mismatch was silently accepted")

