import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines.models import build_model
from LightGenV2.tasks.t19_resnet20_baselines.retrieval import (
    cross_modal_loss, exact_sku_ranking, multi_positive_loss,
)


def test_both_directions_have_identical_architecture_and_parameter_shapes():
    forward = build_model("abo_image_to_text")
    reverse = build_model("abo_text_to_image")
    assert type(forward) is type(reverse)
    assert {k: tuple(v.shape) for k, v in forward.state_dict().items()} == {
        k: tuple(v.shape) for k, v in reverse.state_dict().items()}
    # Separate instances, not shared parameters or an accidental second visual model.
    assert not ({id(p) for p in forward.parameters()} & {id(p) for p in reverse.parameters()})


def test_two_positives_are_both_attracted_not_false_negatives():
    logits = torch.zeros(1, 3, requires_grad=True)
    loss = multi_positive_loss(logits, torch.tensor([[True, True, False]]))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0
    assert logits.grad[0, 2] > 0


def test_reverse_is_transpose_with_same_sku_multi_positive_targets():
    images = torch.tensor([[1., 0.], [0.8, 0.2], [0., 1.]], requires_grad=True)
    titles = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
    image_skus = ["A", "A", "B"]
    title_skus = ["A", "B", "absent_this_batch"]
    actual = cross_modal_loss(images, titles, image_skus, title_skus,
                             direction="text_to_image")
    logits = torch.nn.functional.normalize(images, dim=1) @ torch.nn.functional.normalize(titles, dim=1).T / 0.07
    expected = multi_positive_loss(logits.T[:2], torch.tensor([[True, True, False], [False, False, True]]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
    assert torch.isfinite(titles.grad).all() and titles.grad.abs().sum() > 0


def test_joint_loss_is_mean_of_direction_losses():
    images, titles = torch.randn(4, 8), torch.randn(2, 8)
    args = (images, titles, ["A", "A", "B", "B"], ["A", "B"])
    forward = cross_modal_loss(*args, direction="image_to_text")
    reverse = cross_modal_loss(*args, direction="text_to_image")
    joint = cross_modal_loss(*args, direction="symmetric")
    torch.testing.assert_close(joint, (forward + reverse) / 2)


def test_text_queries_not_multiplied_by_positive_gallery_views():
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0], [0.9, 0.8, 0.7, 0.6]])
    report = exact_sku_ranking(scores, ["A", "B"], ["A", "A", "B", "B"])
    assert report["queries"] == 2 and report["candidates"] == 4
    assert report["first_positive_rank"] == [1, 3]
    assert report["hit_at_1"] == 0.5
    assert report["mrr"] == pytest.approx((1 + 1 / 3) / 2)


def test_ties_preserve_original_candidate_order():
    report = exact_sku_ranking(torch.zeros(1, 3), ["B"], ["A", "B", "B"])
    assert report["order"] == [[0, 1, 2]]
    assert report["first_positive_rank"] == [2]


def test_missing_positive_and_invalid_direction_rejected():
    with pytest.raises(ValueError, match="positive"):
        exact_sku_ranking(torch.zeros(1, 2), ["A"], ["B", "B"])
    with pytest.raises(ValueError, match="direction"):
        cross_modal_loss(torch.ones(1, 2), torch.ones(1, 2), ["A"], ["A"], direction="classifier")
