import json
from pathlib import Path

import numpy as np
import torch

from LightGenV2.tasks.t13_four_modal_lifelong.data import (
    PhysicalRank10Fields, PhysicalTextFields, SpeechRank8Fields,
    _feature_only_field, _feature_text_field, _rgb_field,
)
from LightGenV2.tasks.t13_four_modal_lifelong.model import CrossModalOptics


def test_rgb_uses_all_channels_and_has_unit_power():
    x = np.zeros((2, 56, 56, 3), np.uint8)
    x[0, :, :, 0] = 255
    x[1, :, :, 2] = 255
    field = _rgb_field(x)
    assert field.shape == (2, 224, 224)
    assert not torch.equal(field[0], field[1])
    assert torch.allclose(field.square().sum((-2, -1)), torch.ones(2), atol=1e-5)


def test_physical_video_text_is_balanced_and_uses_both_modalities(tmp_path: Path):
    fields = np.zeros((2, 224, 224), np.float16)
    fields[0, :112] = 1
    fields[1, 112:] = 1
    np.savez(tmp_path / "train.npz", fields=fields, labels=np.array([0, 1]))
    dataset = PhysicalTextFields(tmp_path / "train.npz")
    assert dataset.labels().tolist() == [1, 0, 0, 1]
    batch = dataset[np.arange(4)]
    assert batch.shape == (4, 224, 224)
    assert torch.allclose(batch.square().sum((-2, -1)), torch.ones(4), atol=1e-4)
    assert not torch.equal(batch[0, :, 112:], batch[1, :, 112:])
    assert dataset[:2].shape == (2, 224, 224)


def test_physical_delta_has_no_learned_frontend(tmp_path: Path):
    frames = torch.arange(8, dtype=torch.float32)[:, None, None].expand(8, 112, 56)
    mosaic = frames.reshape(2, 4, 112, 56).permute(0, 2, 1, 3).reshape(224, 224)
    np.savez(tmp_path / "train.npz", fields=mosaic[None].numpy(), labels=np.array([1]))
    dataset = PhysicalTextFields(tmp_path / "train.npz", temporal_delta=True)
    field = dataset[0]
    assert field.shape == (224, 224)
    assert torch.isfinite(field).all()
    assert torch.allclose(field.square().sum(), torch.tensor(1.0), atol=1e-4)


def test_speech_rank8_is_one_example_per_clip(tmp_path: Path):
    images = np.zeros((2, 64, 101, 3), np.uint8)
    images[0, :, :, :] = 32
    images[1, :, :, :] = 192
    np.savez(tmp_path / "train_images.npz", images=images)
    rows = []
    for i, target in enumerate((2, 7)):
        for label, query in ((1, target), (0, (target + 1) % 8)):
            rows.append({"image_local": i, "image_id": f"a{i}", "speaker": f"s{i}",
                         "audio_class": target, "query_class": query, "label": label})
    (tmp_path / "train_questions.json").write_text(json.dumps(rows))
    dataset = SpeechRank8Fields(tmp_path, "train")
    assert len(dataset) == 2
    assert [row["audio_class"] for row in dataset.rows] == [2, 7]
    assert dataset[:].shape == (2, 224, 224)
    assert torch.allclose(dataset[:].square().sum((-2, -1)), torch.ones(2), atol=1e-4)


def test_physical_rank10_uses_all_five_complete_concepts(tmp_path: Path):
    for concept_index, concept in enumerate(PhysicalRank10Fields.CONCEPTS):
        root = tmp_path / concept; root.mkdir()
        field = np.zeros((2, 224, 224), np.float16)
        field[:, :, concept_index:concept_index + 1] = 1
        np.savez(root / "train.npz", fields=field, labels=np.array([0, 1]))
        (root / "train_records.json").write_text(json.dumps([
            {"quadruplet": f"{concept}:0", "label": 0},
            {"quadruplet": f"{concept}:1", "label": 1},
        ]))
    roots = {name: str(tmp_path / name) for name in PhysicalRank10Fields.CONCEPTS}
    dataset = PhysicalRank10Fields(roots, "train")
    assert len(dataset) == 10
    assert dataset.labels().tolist() == list(range(10))
    assert dataset[:].shape == (10, 224, 224)
    assert torch.allclose(dataset[:].square().sum((-2, -1)), torch.ones(10), atol=1e-4)


def test_frozen_feature_encodings_preserve_power_and_text():
    features = np.ones((2, 128), np.float16)
    features[1, 0] = 4
    plain = _feature_only_field(features)
    assert plain.shape == (2, 224, 224)
    assert torch.allclose(plain.square().sum((-2, -1)), torch.ones(2), atol=1e-5)
    tokens = np.zeros((2, 32), np.uint8)
    tokens[:, :2] = [[2, 3], [2, 4]]
    paired = _feature_text_field(features, tokens)
    assert paired.shape == (2, 224, 224)
    assert torch.allclose(paired.square().sum((-2, -1)), torch.ones(2), atol=1e-5)
    assert not torch.equal(paired[0], paired[1])


def test_single_task_and_lifelong_geometries_are_explicit():
    compact = CrossModalOptics("moe", max_experts=4, optical_layers=2)
    lifelong = CrossModalOptics("moe", max_experts=16, optical_layers=2)
    assert (compact.height, compact.active_height, len(compact.first_phase)) == (518, 478, 4)
    assert (lifelong.height, lifelong.active_height, len(lifelong.first_phase)) == (1026, 986, 16)
    assert isinstance(compact.heads["speech"], torch.nn.Linear)
    assert compact.heads["speech"].out_features == 8
    assert compact.heads["physical"].out_features == 10
    assert sum(isinstance(module, torch.nn.Linear)
               for module in compact.heads["speech"].modules()) == 1
