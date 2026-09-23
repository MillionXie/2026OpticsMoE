import pytest
import torch

from LightGenV2.tasks.t08_abo_image_text_retrieval.abo_backbone_baselines import (
    DualRetrievalReadout,
    _candidate_metrics,
    _gallery_metrics,
    _limit_per_key,
    _symmetric_prototype_loss,
)


def test_candidate_metrics_exact_ranking():
    scores = torch.tensor([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]])
    metrics, order = _candidate_metrics(scores, torch.tensor([0, 1, 2]))
    assert metrics["r_at_1"] == pytest.approx(2 / 3)
    assert metrics["r_at_5"] == 1.0
    assert order.shape == (3, 3)


def test_gallery_metrics_multiple_positives():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 4.0, 3.0]])
    metrics, order = _gallery_metrics(scores, torch.tensor([0, 1]), torch.tensor([0, 0, 1, 1]))
    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["map"] == 1.0
    assert order.shape == (2, 4)


def test_limit_per_key_is_stable():
    rows = [{"key": "a", "v": 1}, {"key": "a", "v": 2}, {"key": "b", "v": 3}]
    assert _limit_per_key(rows, lambda row: row["key"], 1) == [rows[0], rows[2]]
    assert _limit_per_key(rows, lambda row: row["key"], 0) is rows


def test_dual_readout_outputs_normalized_shared_space():
    head = DualRetrievalReadout(8, 6, 4)
    images = head.encode_images(torch.randn(5, 8))
    titles = head.encode_texts(torch.randn(3, 6))
    assert images.shape == (5, 4)
    assert titles.shape == (3, 4)
    assert torch.allclose(torch.linalg.vector_norm(images, dim=1), torch.ones(5))
    assert torch.allclose(torch.linalg.vector_norm(titles, dim=1), torch.ones(3))


def test_symmetric_prototype_loss_is_finite():
    images = torch.nn.functional.normalize(torch.randn(6, 4), dim=1)
    titles = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = _symmetric_prototype_loss(images, labels, titles, 0.07)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
