from __future__ import annotations

from types import SimpleNamespace

import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import (
    Modality,
    OpticalRetrieval,
)


def test_noise_ablation_modes_select_only_declared_feature_layers() -> None:
    model = OpticalRetrieval.__new__(OpticalRetrieval)
    torch.nn.Module.__init__(model)
    model.vision = SimpleNamespace(optical_noise_blocks=frozenset())
    model.language = SimpleNamespace(optical_noise_blocks=frozenset())

    model.set_optical_noise_ablation("language_global")
    assert model.vision.optical_noise_blocks == frozenset()
    assert model.language.optical_noise_blocks == frozenset({2})

    model.set_optical_noise_ablation("global_each_modality")
    assert model.vision.optical_noise_blocks == frozenset({2})
    assert model.language.optical_noise_blocks == frozenset({2})

    model.set_optical_noise_ablation("all_feature_layers")
    assert model.vision.optical_noise_blocks == frozenset({1, 2})
    assert model.language.optical_noise_blocks == frozenset({1, 2})


def test_selected_optical_output_is_replaced_by_seeded_noise() -> None:
    modality = Modality.__new__(Modality)
    torch.nn.Module.__init__(modality)
    modality.optical_noise_blocks = frozenset({2})
    value = torch.full((2, 3, 4), 7.0)
    assert torch.equal(modality._ablate_optical_output(value, 1), value)
    torch.manual_seed(42)
    first = modality._ablate_optical_output(value, 2)
    torch.manual_seed(42)
    second = modality._ablate_optical_output(value, 2)
    assert torch.equal(first, second)
    assert not torch.equal(first, value)

