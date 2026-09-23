import os
from pathlib import Path

import numpy as np
import pytest
import torch

from LightGenV2.tasks.t14_shared_readout_lifelong.data import (
    ClevrRawPairs, PhysicalPermutedCandidates, SpeechPermutedCandidates,
    candidate_permutations,
    position_labels,
)


def test_candidate_position_is_label_and_permutation_is_reproducible():
    order = candidate_permutations(64, 8, 17)
    labels = np.zeros(64, dtype=np.int64)
    positions = position_labels(order, labels)
    assert np.array_equal(order[np.arange(64), positions], labels)
    assert len(np.unique(positions)) > 1
    assert np.array_equal(order, candidate_permutations(64, 8, 17))
    with pytest.raises(ValueError):
        position_labels(np.zeros((2, 8), dtype=np.int16), labels[:2])


def test_clevr_uses_one_positive_and_one_negative_query_per_original_image(tmp_path: Path):
    split = "train"
    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(3)[:, None, None] * 70
    np.save(tmp_path / f"{split}_images.npy", images)
    np.save(tmp_path / f"{split}_image_index.npy", np.repeat(np.arange(3), 6))
    np.save(tmp_path / f"{split}_labels.npy", np.tile([1, 0, 1, 0, 1, 0], 3))
    np.save(tmp_path / f"{split}_token_ids.npy", np.tile(
        np.arange(32, dtype=np.uint8), (18, 1)))
    fields = ClevrRawPairs(tmp_path, split)
    assert len(fields) == 6
    assert fields.labels.tolist() == [1, 0] * 3
    assert fields.image_index.tolist() == [0, 0, 1, 1, 2, 2]
    batch = fields.get_batch([0, 2], torch.device("cpu"))
    assert tuple(batch.shape) == (2, 224, 224)
    assert torch.isfinite(batch).all()


def test_full_clevr_source_retains_every_image_when_available():
    source = os.environ.get("T14_CLEVR_SOURCE")
    if not source:
        pytest.skip("set T14_CLEVR_SOURCE for the lab-data integration test")
    for split, expected_images in (("train", 70000), ("val", 7500),
                                   ("test", 7500)):
        fields = ClevrRawPairs(Path(source), split)
        assert len(fields.images) == expected_images
        assert len(fields) == 2 * expected_images
        assert np.bincount(fields.labels).tolist() == [expected_images,
                                                        expected_images]


def test_speech_text_order_changes_the_correct_output_position():
    bank = torch.arange(8 * 28 * 112).reshape(8, 28, 112)
    dataset = object.__new__(SpeechPermutedCandidates)
    class FakeSpeech:
        def __getitem__(self, indices):
            return torch.zeros(len(indices), 224, 224)
    dataset.base = FakeSpeech()
    dataset.original_bank = bank
    dataset.permutations = np.stack((np.arange(8), np.roll(np.arange(8), 1)))
    dataset.original_labels = np.array([0, 0])
    dataset.labels = position_labels(dataset.permutations, dataset.original_labels)
    fields = dataset.get_batch([0, 1], torch.device("cpu"))
    assert dataset.labels.tolist() == [0, 1]
    assert tuple(fields.shape) == (2, 224, 224)
    assert torch.equal(fields[0, :28, 112:], bank[0])
    assert torch.equal(fields[1, 28:56, 112:], bank[0])


def test_full_speech_source_has_variable_correct_positions_when_available():
    source = os.environ.get("T14_SPEECH_SOURCE")
    if not source:
        pytest.skip("set T14_SPEECH_SOURCE for the lab-data integration test")
    for split, expected in (("train", 6263), ("val", 843), ("test", 867)):
        fields = SpeechPermutedCandidates(Path(source), split)
        assert len(fields) == expected
        assert np.array_equal(fields.permutations[
            np.arange(expected), fields.labels], fields.original_labels)
        assert len(np.unique(fields.labels)) == 8
    batch = fields.get_batch([0, 1], torch.device("cpu"))
    assert tuple(batch.shape) == (2, 224, 224)
    assert not torch.equal(batch[0, :, 112:], batch[1, :, 112:])


def test_physical_caption_order_changes_the_correct_output_position():
    dataset = object.__new__(PhysicalPermutedCandidates)
    class FakePhysical:
        def __getitem__(self, indices):
            return torch.zeros(len(indices), 224, 224)
    dataset.base = FakePhysical()
    dataset.permutations = np.stack((np.arange(10), np.roll(np.arange(10), 1)))
    dataset.original_labels = np.array([0, 0])
    dataset.labels = position_labels(dataset.permutations, dataset.original_labels)
    edges = np.linspace(0, 224, 11).round().astype(int)
    dataset.bands = [torch.arange(1, 11, dtype=torch.float32)[:, None, None].expand(
        10, int(edges[pos + 1] - edges[pos]), 112).clone()
        for pos in range(10)]
    fields = dataset.get_batch([0, 1], torch.device("cpu"))
    assert dataset.labels.tolist() == [0, 1]
    assert tuple(fields.shape) == (2, 224, 224)
    assert fields[0, 0, 112] < fields[1, 0, 112]
    assert fields[0, 25, 112] > fields[1, 25, 112]
