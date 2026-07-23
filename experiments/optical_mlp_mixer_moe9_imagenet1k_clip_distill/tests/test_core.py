from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import (
    EpochViewSampler,
    HuggingFaceImageNetViewDataset,
    view_seed,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.optics.geometry import (
    MoEGeometry,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.optics.mixer import (
    FoldedOpticalMixerBlock,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.optics.physical import (
    ExpertSquareDetectionReload,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.settings import (
    ExperimentSettings,
    load_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def tiny_settings() -> ExperimentSettings:
    settings = ExperimentSettings()
    settings.model.image_size = 4
    settings.model.patch_size = 2
    settings.model.token_count = 4
    settings.model.hidden_size = 8
    settings.model.clip_projection_dim = 8
    settings.model.num_classes = 4
    settings.geometry.expert_size = 8
    settings.geometry.expert_pitch = 38
    settings.geometry.active_size = 114
    settings.geometry.canvas_size = 116
    settings.geometry.outer_padding_per_side = 1
    settings.router.pool_size = 4
    settings.optics.inter_layer_distance_m = 1e-3
    settings.optics.readout_to_global_distance_m = 1e-3
    settings.optics.global_to_detector_distance_m = 1e-3
    return settings


def test_formal_config_and_parameter_formula() -> None:
    settings = load_settings(ROOT / "configs" / "imagenet1k.json")
    assert settings.model.name == "OpticalMixerMoE9"
    assert settings.dataset.source == "huggingface"
    assert settings.dataset.download is True
    assert settings.dataset.hf_dataset_id == "ILSVRC/imagenet-1k"
    assert settings.dataset.validation_split == "validation"
    assert settings.model.num_blocks == 7
    assert settings.training.epochs == 90
    assert settings.optimizer.name == "adamw"
    assert settings.optimizer.weight_decay == 0.0
    assert settings.optical_parameter_formula == {
        "phase_parameters_per_expert_plane": 451_584,
        "expert_phase_parameters_per_block": 2_257_920,
        "global_phase_parameters_per_block": 580_644,
        "optical_phase_parameters_per_block": 2_838_564,
        "expert_phase_parameters_total": 15_805_440,
        "global_phase_parameters_total": 4_064_508,
        "optical_phase_parameters_total": 19_869_948,
    }


def test_new_geometry_preserves_gap_padding_and_centered_detector() -> None:
    geometry = MoEGeometry(792, 762, 224, 254, 9)
    geometry.validate()
    assert geometry.active_start == 15
    assert geometry.expert_pitch - geometry.expert_size == 30
    assert geometry.expert_apertures[0].y0 == 30
    assert geometry.expert_apertures[-1].y1 == 762
    assert geometry.detector_aperture == geometry.expert_apertures[4]
    assert (
        geometry.detector_aperture.y0,
        geometry.detector_aperture.y1,
    ) == (284, 508)


def test_token_and_channel_mapping_are_zero_padded_without_interpolation() -> None:
    block = FoldedOpticalMixerBlock(tiny_settings())
    hidden = torch.randn(2, 4, 8)
    token = block._token_field(hidden)
    channel = block._channel_field(hidden)
    assert token.shape == (2, 8, 8)
    assert channel.shape == (2, 8, 8)
    assert torch.all(token[:, :, 4:] == 0)
    assert torch.all(channel[:, 4:, :] == 0)
    assert torch.all(token[:, :, :4] > 0)
    assert torch.all(channel[:, :4, :] > 0)
    source = inspect.getsource(block._token_field) + inspect.getsource(block._channel_field)
    assert "interpolate" not in source


def test_block_routes_once_and_preserves_shape_and_gradients() -> None:
    block = FoldedOpticalMixerBlock(tiny_settings())
    calls = []
    hook = block.core.router.register_forward_hook(lambda *_: calls.append(1))
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    output = block(hidden)
    hook.remove()
    assert output.shape == hidden.shape
    assert len(calls) == 1
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    phase_gradients = [
        parameter.grad
        for name, parameter in block.named_parameters()
        if name.endswith("raw_phase")
    ]
    assert phase_gradients
    assert all(value is not None for value in phase_gradients)
    assert all(torch.isfinite(value).all() for value in phase_gradients)


def test_oeo_output_is_nonnegative_and_unselected_experts_are_zero() -> None:
    geometry = MoEGeometry(116, 114, 8, 38, 9)
    conversion = ExpertSquareDetectionReload(
        geometry.canvas_size,
        geometry.expert_apertures,
        eps=1e-5,
        nonlinearity="relu",
        per_expert_enabled=True,
        elementwise_affine=False,
    )
    field = torch.complex(torch.randn(2, 116, 116), torch.randn(2, 116, 116))
    selected = torch.zeros(2, 9, dtype=torch.bool)
    selected[:, :3] = True
    weights = selected.float() / 3
    output = conversion(
        field, selected_experts=selected, routing_weights=weights
    )
    assert output.dtype == torch.complex64
    assert torch.all(output.real >= 0)
    assert torch.all(output.imag == 0)
    for aperture in geometry.expert_apertures[3:]:
        assert torch.count_nonzero(
            output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
        ) == 0


def test_epoch_sampler_covers_every_base_image_and_cycles_views() -> None:
    class FakeDataset:
        base_sample_count = 11
        views = 4

    sampler = EpochViewSampler(FakeDataset(), shuffle=False, seed=42)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert sorted(index // 4 for index in first) == list(range(11))
    assert sorted(index // 4 for index in second) == list(range(11))
    assert all((b % 4) == ((a % 4) + 1) % 4 for a, b in zip(first, second))
    assert view_seed(42, 3, 1) == view_seed(42, 3, 1)
    assert view_seed(42, 3, 1) != view_seed(42, 3, 2)


def test_huggingface_backend_preserves_labels_and_returns_deterministic_view() -> None:
    class FakeLabelFeature:
        names = ["zero", "one", "two", "three"]

    class FakeDataset:
        features = {"label": FakeLabelFeature()}
        _fingerprint = "fake-imagenet-fingerprint"

        def __init__(self) -> None:
            self.records = [
                {"image": Image.new("RGB", (8, 8), color=(index * 30, 10, 20)), "label": index}
                for index in range(4)
            ]

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index):
            if index == "label":
                return [item["label"] for item in self.records]
            return self.records[index]

    settings = tiny_settings()
    settings.model.num_classes = 4
    dataset = HuggingFaceImageNetViewDataset(
        FakeDataset(),
        settings,
        split="validation",
        train=False,
        limit=None,
    )
    first = dataset[0]
    second = dataset[0]
    assert dataset.base_sample_count == 4
    assert dataset.classes == ["zero", "one", "two", "three"]
    assert first["image"].shape == (3, 4, 4)
    assert first["label"] == 0
    assert first["path"].startswith("hf://ILSVRC/imagenet-1k/validation/")
    assert torch.equal(first["image"], second["image"])


def test_config_json_is_pretty_and_only_requested_model_exists() -> None:
    path = ROOT / "configs" / "imagenet1k.json"
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert "\n  \"dataset\"" in text
    assert parsed["model"]["name"] == "OpticalMixerMoE9"
    assert "sam" not in parsed
    assert "fine_tuning" not in parsed
