from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone.initialization import (
    InitializationContractError,
    initialize_from_frozen_p11_backbone,
    sha256_file,
    state_dict_sha256,
)


class FakeStem(nn.Module):
    def __init__(self, digest: str) -> None:
        super().__init__()
        self.checkpoint_sha256 = digest


class FakeModel(nn.Module):
    def __init__(self, stem_digest: str) -> None:
        super().__init__()
        self.stem = FakeStem(stem_digest)
        self.adapter = nn.Linear(3, 4)
        self.readout = nn.Linear(4, 7)


def _checkpoint(path: Path, model: FakeModel, stem_digest: str, *, leak_head: bool = False) -> str:
    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if leak_head or not name.startswith("readout.")
    }
    torch.save({"backbone": state, "stem_checkpoint_sha256": stem_digest}, path)
    return sha256_file(path)


def test_strict_backbone_load_leaves_new_head_untouched(tmp_path: Path) -> None:
    stem_digest = "a" * 64
    source = FakeModel(stem_digest)
    path = tmp_path / "backbone.pt"
    digest = _checkpoint(path, source, stem_digest)
    target = FakeModel(stem_digest)
    head_before = state_dict_sha256(
        {name: value for name, value in target.state_dict().items() if name.startswith("readout.")}
    )
    report = initialize_from_frozen_p11_backbone(
        target,
        backbone_checkpoint=path,
        expected_backbone_sha256=digest,
        expected_stem_sha256=stem_digest,
    )
    assert report["copied_imagenet1k_readout"] is False
    assert report["missing_keys"] == ["readout.bias", "readout.weight"]
    assert report["new_readout_state_sha256"] == head_before


def test_source_readout_is_rejected(tmp_path: Path) -> None:
    stem_digest = "b" * 64
    model = FakeModel(stem_digest)
    path = tmp_path / "bad.pt"
    digest = _checkpoint(path, model, stem_digest, leak_head=True)
    with pytest.raises(InitializationContractError, match="readout"):
        initialize_from_frozen_p11_backbone(
            FakeModel(stem_digest),
            backbone_checkpoint=path,
            expected_backbone_sha256=digest,
            expected_stem_sha256=stem_digest,
        )
