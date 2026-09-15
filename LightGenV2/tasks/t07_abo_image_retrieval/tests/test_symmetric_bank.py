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


def test_top1_online_matches_shared_objective_both_views_and_frozen_bank():
    from torch.nn import functional as F
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import train_ranking_loss
    z, labels, bank, bl, ex=fixture()
    sim=F.normalize(z.float(),dim=1)@F.normalize(bank.detach().float(),dim=1).T/.1
    pos=labels[:,None].eq(bl[None]) & ~ex
    expected=train_ranking_loss(sim,pos,ex,'top1_softplus')
    loss,hit=paired_bank_loss(z,labels,bank,2,bl,ex,symmetric=True,supcon_weight=0,ranking_loss='top1_softplus')
    assert torch.allclose(loss,expected) and 0<=hit<=1
    loss.backward()
    assert torch.isfinite(z.grad).all() and (z.grad.norm(dim=1)>0).all()
    assert bank.grad is None


def test_online_top1_rejects_missing_positive_or_unknown_objective():
    z,labels,bank,bl,ex=fixture()
    with pytest.raises(ValueError,match='Unknown'):
        paired_bank_loss(z,labels,bank,2,bl,ex,symmetric=True,ranking_loss='bad')
    ex[3]=True
    with pytest.raises(ValueError,match='nonself'):
        paired_bank_loss(z,labels,bank,2,bl,ex,symmetric=True,ranking_loss='top1_softplus')


def test_two_view_joint_profile_has_no_inference_expansion():
    p=dict(PROFILES['sku_two_view_joint'])
    assert p.pop('ranking_loss')=='two_view_softplus'
    assert p.pop('symmetric_bank') is True and p.pop('supcon_weight')==0
    assert p['positive_weight']==0
    p['positive_weight']=.1
    assert p==PROFILES['sku_capacity_control']


def test_two_view_online_both_query_views_match_shared_loss():
    from torch.nn import functional as F
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import train_ranking_loss
    torch.manual_seed(58)
    z=torch.randn(4,8,requires_grad=True); bank=torch.randn(6,8,requires_grad=True)
    labels=torch.tensor([0,1,0,1]); bl=torch.tensor([0,0,0,1,1,1])
    ex=torch.eye(6,dtype=torch.bool)[[0,3,1,4]]
    sim=F.normalize(z,dim=1)@F.normalize(bank.detach(),dim=1).T/.1
    expected=train_ranking_loss(sim,labels[:,None].eq(bl[None]),ex,'two_view_softplus')
    loss,_=paired_bank_loss(z,labels,bank,2,bl,ex,symmetric=True,supcon_weight=0,ranking_loss='two_view_softplus')
    assert torch.allclose(loss,expected)
    loss.backward()
    assert bank.grad is None and (z.grad.norm(dim=1)>0).all() and torch.isfinite(z.grad).all()
