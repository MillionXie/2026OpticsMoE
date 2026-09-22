import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from LightGenV2.tasks.t13_four_modal_lifelong.data import (
    FeatureFields, PhysicalRank10Fields, PhysicalTextFields, SpeechRank8Fields,
    _feature_only_field, _feature_text_field, _rgb_field,
)
from LightGenV2.tasks.t13_four_modal_lifelong.model import CrossModalOptics
from LightGenV2.tasks.t13_four_modal_lifelong import run as experiment_run
from LightGenV2.tasks.t13_four_modal_lifelong.fit_cross_task_d2nn import parse_checkpoints
from LightGenV2.tasks.t13_four_modal_lifelong.prepare_physical_probe import encode_batch
from LightGenV2.tasks.t12_cross_modal_lifelong.prepare_physical_concepts import encode_video


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


def test_vectorized_physical_encoder_matches_reference():
    video = np.random.default_rng(17).integers(0, 256, (2, 15, 64, 64, 3), dtype=np.uint8)
    actual = encode_batch(video)
    expected = np.stack([encode_video(item) for item in video])
    assert actual.shape == (2, 224, 224)
    assert np.allclose(actual, expected, atol=2e-3)


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


def test_feature_fields_device_expansion_matches_reference(tmp_path: Path):
    features = np.arange(256, dtype=np.float16).reshape(2, 128)
    tokens = np.zeros((2, 32), np.uint8)
    tokens[:, :2] = [[2, 3], [4, 5]]
    np.save(tmp_path / "train_features.npy", features)
    np.save(tmp_path / "train_token_ids.npy", tokens)
    fields = FeatureFields(tmp_path, "train", with_text=True)
    expected = fields[:]
    actual = fields.get_batch(slice(None), torch.device("cpu"))
    assert torch.allclose(actual, expected)


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


def test_head_refit_is_rejected_when_validation_score_decreases(monkeypatch):
    model = SimpleNamespace(heads={"speech": torch.nn.Linear(3, 8)})
    original = {key: value.detach().clone()
                for key, value in model.heads["speech"].state_dict().items()}
    task = SimpleNamespace(name="speech")
    scores = iter((.80, .70))

    def fake_evaluate(*_args, **_kwargs):
        return ({"balanced_accuracy": next(scores)}, None, None)

    def fake_fit(*_args, **_kwargs):
        head = torch.nn.Linear(3, 8)
        with torch.no_grad():
            head.weight.fill_(99)
            head.bias.fill_(99)
        return head, {"selected_epoch": 1, "metrics": {}}

    monkeypatch.setattr(experiment_run, "evaluate", fake_evaluate)
    monkeypatch.setattr(experiment_run, "fit_mlp_head", fake_fit)
    result = experiment_run.calibrate_head(
        model, task, {"eval_batch": 4}, torch.device("cpu"), seed=17)

    assert result["accepted"] is False
    assert result["validation_score_before"] == .80
    assert result["validation_score_after"] == .70
    for key, value in model.heads["speech"].state_dict().items():
        assert torch.equal(value, original[key])


def test_resume_restores_only_fully_evaluated_stages(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    expected = {key: torch.full_like(value, 3) for key, value in model.state_dict().items()}
    stage1 = tmp_path / "stage_1_eurosat"
    stage1.mkdir()
    torch.save({"model": expected}, stage1 / "best_checkpoint.pt")
    (stage1 / "stage_result.json").write_text(json.dumps({
        "selected_epoch": 7,
        "val": {"eurosat": {"balanced_accuracy": .8}},
        "test": {"eurosat": {"balanced_accuracy": .79}},
    }))
    incomplete = tmp_path / "stage_2_clevr"
    incomplete.mkdir()
    torch.save({"model": model.state_dict()}, incomplete / "best_checkpoint.pt")

    start, history, replay = experiment_run.restore_completed_stages(
        model, tmp_path, use_replay=True)

    assert start == 1
    assert history == [{
        "task": "eurosat", "selected_epoch": 7,
        "val": {"eurosat": {"balanced_accuracy": .8}},
        "test": {"eurosat": {"balanced_accuracy": .79}},
    }]
    assert replay == {"eurosat": None}
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_continual_matrix_accepts_stage_evaluation_val_key(tmp_path: Path):
    history = [
        {
            "task": "eurosat",
            "val": {"eurosat": {"balanced_accuracy": .80}},
            "test": {"eurosat": {"balanced_accuracy": .79}},
        },
        {
            "task": "clevr",
            "val": {
                "eurosat": {"balanced_accuracy": .60},
                "clevr": {"balanced_accuracy": .75},
            },
            "test": {
                "eurosat": {"balanced_accuracy": .59},
                "clevr": {"balanced_accuracy": .74},
            },
        },
    ]

    result = experiment_run.save_continual_matrix(tmp_path, history)

    assert result["validation"] == [
        {"eurosat": .80},
        {"eurosat": .60, "clevr": .75},
    ]
    assert result["test"] == [
        {"eurosat": .79},
        {"eurosat": .59, "clevr": .74},
    ]
    assert result["continual"]["forgetting"]["eurosat"] == pytest.approx(.20)


def test_formal_cross_task_probe_requires_all_four_checkpoints():
    parsed = parse_checkpoints([f"{name}=/{name}.pt" for name in experiment_run.TASK_ORDER])
    assert tuple(parsed) == experiment_run.TASK_ORDER
    assert parsed["speech"] == Path("/speech.pt")
