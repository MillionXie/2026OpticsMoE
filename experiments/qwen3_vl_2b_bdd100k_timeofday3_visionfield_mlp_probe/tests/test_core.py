from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from ..feature_probe import load_feature_cache
from ..modeling import VisionFieldProbeHead
from ..optics.stacks import VisionOpticalStackSurrogate
from ..training import train_probe


def _encoder() -> VisionOpticalStackSurrogate:
    return VisionOpticalStackSurrogate(
        hidden_size=1024, optical_dim=64, conversions=4, field_size=64, padding_size=128,
        wavelength_nm=532.0, pixel_pitch_um=8.0, distance_cm=5.0,
        amplitude_mask_enabled=False, phase_init="zeros", phase_init_std=0.02,
    )


def test_encode_input_field_shape_nonnegative_and_skips_conversions(monkeypatch):
    encoder = _encoder()
    for conversion in encoder.conversions:
        monkeypatch.setattr(conversion, "forward", lambda _: (_ for _ in ()).throw(AssertionError("conversion ran")))
    fields = encoder.encode_groups_to_input_fields([torch.randn(60, 1024)])
    assert fields.shape == (1, 64, 64)
    assert fields.flatten(1).shape == (1, 4096)
    assert torch.all(fields >= 0)
    assert torch.count_nonzero(fields[:, 60:]) == 0


def test_visual_token_overflow_is_explicit():
    with pytest.raises(RuntimeError, match=r"visual token count 65 exceeds optical_field_size=64"):
        _encoder().encode_groups_to_input_fields([torch.randn(65, 1024)])


@pytest.mark.parametrize("head_type", ["linear", "mlp", "bottleneck"])
def test_probe_head_shape_and_backward(head_type):
    head = VisionFieldProbeHead(4096, 3, head_type, 128, 0.1)
    logits = head(torch.randn(4, 4096))
    assert logits.shape == (4, 3)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_default_mlp_parameter_count():
    head = VisionFieldProbeHead(4096, 3, "mlp", 512, 0.1)
    assert sum(parameter.numel() for parameter in head.parameters()) == 2_099_203


def test_feature_cache_round_trip(tmp_path):
    path = tmp_path / "cache.pt"
    payload = {
        "features": torch.randn(6, 4096).half(), "labels": torch.tensor([0, 1, 2, 0, 1, 2]),
        "sample_indices": torch.arange(6), "image_grid_thw": torch.ones(6, 3, dtype=torch.long),
        "visual_token_counts": torch.full((6,), 60), "class_names": ["daytime", "night", "dawn_dusk"],
        "metadata": {"feature_type": "vision_optical_input_field"},
    }
    torch.save(payload, path)
    loaded = load_feature_cache(path)
    assert torch.equal(loaded["features"], payload["features"])
    assert torch.equal(loaded["labels"], payload["labels"])
    assert loaded["metadata"] == payload["metadata"]


def test_one_epoch_fixed_feature_training_writes_outputs(tmp_path):
    output = tmp_path / "run"; (output / "features").mkdir(parents=True)
    labels = torch.tensor([0, 1, 2] * 6)
    torch.save({
        "features": torch.randn(len(labels), 4096).half(), "labels": labels,
        "sample_indices": torch.arange(len(labels)), "image_grid_thw": torch.ones(len(labels), 3, dtype=torch.long),
        "visual_token_counts": torch.full((len(labels),), 60),
        "class_names": ["daytime", "night", "dawn_dusk"], "metadata": {},
    }, output / "features" / "train_vision_input_field.pt")
    settings = SimpleNamespace(
        output_dir=output, optical_field_size=64, validation_fraction=0.2, seed=42,
        head_batch_size=4, num_workers=0, probe_head_type="linear", probe_hidden_dim=128,
        probe_dropout=0.1, learning_rate=1e-3, weight_decay=0.0, epochs=1,
    )
    train_probe(settings, ["daytime", "night", "dawn_dusk"], torch.device("cpu"))
    assert (output / "metrics" / "probe_training_history.csv").is_file()
    assert (output / "metrics" / "best_validation.json").is_file()
    assert (output / "checkpoints" / "probe_head_best.pt").is_file()
    assert (output / "checkpoints" / "probe_head_last.pt").is_file()
    report = json.loads((output / "metrics" / "probe_model.json").read_text(encoding="utf-8"))
    assert report["probe_total_trainable_parameters"] == 12_291

