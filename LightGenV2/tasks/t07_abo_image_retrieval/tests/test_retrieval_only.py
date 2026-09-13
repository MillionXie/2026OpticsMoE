import pytest
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import retrieval_loss
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli import supcon


def tensors():
    g = torch.Generator().manual_seed(54)
    return (torch.randn(4, 64, generator=g, requires_grad=True), torch.tensor([0, 1, 0, 1]),
            torch.randn(4, 64, generator=g, requires_grad=True), torch.tensor([0, 0, 1, 1]),
            torch.tensor([[True, False, False, False], [False, False, True, False]]))


def test_default_loss_keeps_previous_formula():
    z, labels, bank, bank_labels, excluded = tensors()
    old = retrieval_loss(z, labels, bank, 2, bank_labels, excluded)[0]
    pure = retrieval_loss(z, labels, bank, 2, bank_labels, excluded, supcon_weight=0.)[0]
    assert torch.allclose(old, pure + .5 * supcon(F.normalize(z.float(), dim=-1), labels))


def test_pure_retrieval_equals_masked_multi_positive_nll_and_backpropagates():
    z, labels, bank, bank_labels, excluded = tensors()
    loss, hit = retrieval_loss(z, labels, bank, 2, bank_labels, excluded, supcon_weight=0.)
    logits = F.normalize(z[:2], dim=-1) @ F.normalize(bank.detach(), dim=-1).T / .1
    logits = logits.masked_fill(excluded, -torch.inf)
    positive = labels[:2, None].eq(bank_labels[None]) & ~excluded
    expected = (logits.logsumexp(1) - logits.masked_fill(~positive, -torch.inf).logsumexp(1)).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad[:2].abs().sum() > 0
    assert z.grad[2:].abs().sum() == 0 and bank.grad is None
    assert 0 <= hit <= 1


def test_profile_only_removes_two_view_compaction_losses():
    control, trial = PROFILES['sku_capacity_control'], PROFILES['sku_retrieval_only']
    assert trial['positive_weight'] == trial['supcon_weight'] == 0.
    assert {k: v for k, v in trial.items() if k not in ('positive_weight', 'supcon_weight')} == {
        k: v for k, v in control.items() if k != 'positive_weight'}


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_supcon_weight_rejected(weight):
    z, labels, bank, bank_labels, excluded = tensors()
    with pytest.raises(ValueError, match='SupCon weight'):
        retrieval_loss(z, labels, bank, 2, bank_labels, excluded, supcon_weight=weight)
