from __future__ import annotations

import torch

from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import (
    _compact_residual_mlp_state,
)


def test_compaction_keeps_strongest_paired_hidden_neurons() -> None:
    source = {
        "blocks.0.mlp.0.weight": torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 0.5]]),
        "blocks.0.mlp.0.bias": torch.arange(4.0),
        "blocks.0.mlp.3.weight": torch.tensor([[1.0, 4.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
        "blocks.0.mlp.3.bias": torch.tensor([7.0, 8.0]),
        "unchanged": torch.tensor([9.0]),
    }
    target = {
        "blocks.0.mlp.0.weight": torch.empty(2, 2),
        "blocks.0.mlp.0.bias": torch.empty(2),
        "blocks.0.mlp.3.weight": torch.empty(2, 2),
        "blocks.0.mlp.3.bias": torch.empty(2),
        "unchanged": torch.empty(1),
    }
    compact, report = _compact_residual_mlp_state(source, target)
    # Importance is [1,12,4,.5], so hidden neurons 1 and 2 survive.
    assert report["kept"]["blocks.0.mlp"] == [1, 2]
    assert torch.equal(compact["blocks.0.mlp.0.bias"], torch.tensor([1.0, 2.0]))
    assert torch.equal(compact["blocks.0.mlp.3.weight"], source["blocks.0.mlp.3.weight"][:, [1, 2]])
    assert torch.equal(compact["unchanged"], source["unchanged"])
