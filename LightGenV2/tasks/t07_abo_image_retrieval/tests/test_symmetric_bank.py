import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import paired_bank_loss, retrieval_loss
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES


def fixture():
    torch.manual_seed(23)
    z = torch.randn(4, 8, requires_grad=True)
    bank = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    bank_labels = torch.tensor([0, 0, 1, 1])
    excluded = torch.eye(4, dtype=torch.bool)[[0, 2, 1, 3]]
    return z, labels, bank, bank_labels, excluded


def test_default_matches_original_exactly():
    z, labels, bank, bl, excluded = fixture()
    old = retrieval_loss(z, labels, bank, 2, bl, excluded[:2])
    new = paired_bank_loss(z, labels, bank, 2, bl, excluded[:2])
    assert all(torch.equal(a, b) for a, b in zip(old, new))


def test_symmetric_is_average_not_double_loss_and_both_views_receive_nll():
    z, labels, bank, bl, ex = fixture()
    left = retrieval_loss(z, labels, bank, 2, bl, ex[:2], supcon_weight=0)[0]
    right = retrieval_loss(z[[2, 3, 0, 1]], labels[[2, 3, 0, 1]], bank, 2, bl, ex[2:], supcon_weight=0)[0]
    loss, hit = paired_bank_loss(z, labels, bank, 2, bl, ex, symmetric=True, supcon_weight=0)
    assert torch.allclose(loss, (left + right) / 2)
    assert 0 <= hit <= 1
    loss.backward()
    assert torch.isfinite(z.grad).all() and (z.grad.norm(dim=1) > 0).all()
    assert bank.grad is None


@pytest.mark.parametrize('bad', [None, torch.zeros(2, 4, dtype=torch.bool)])
def test_missing_second_view_self_mask_rejected(bad):
    z, labels, bank, bl, _ = fixture()
    with pytest.raises(ValueError, match='BOTH'):
        paired_bank_loss(z, labels, bank, 2, bl, bad, symmetric=True)


def test_second_view_missing_nonself_positive_rejected():
    z, labels, bank, bl, ex = fixture()
    ex[2] = True
    with pytest.raises(ValueError, match='nonself'):
        paired_bank_loss(z, labels, bank, 2, bl, ex, symmetric=True)


def test_profile_changes_training_only():
    p = dict(PROFILES['sku_symmetric_bank'])
    assert p.pop('symmetric_bank') is True
    assert p == PROFILES['sku_capacity_control']
