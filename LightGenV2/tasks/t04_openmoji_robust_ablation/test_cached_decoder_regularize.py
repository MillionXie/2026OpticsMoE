import torch

from LightGenV2.tasks.t04_openmoji_robust_ablation.lab_cached_decoder_regularize import category_mixup_loss, paired_consistency


def test_consistency_matches_historical_loss_and_detaches_teacher():
    import torch.nn.functional as F
    torch.manual_seed(7)
    cat = torch.randn(2, 4, 3, 3, requires_grad=True)
    edit = torch.randn(2, 3, 3, requires_grad=True)
    nc = torch.randn(2, 4, 3, 3, requires_grad=True)
    ne = torch.randn(2, 3, 3, requires_grad=True)
    expected = F.kl_div(nc.log_softmax(1), cat.detach().softmax(1), reduction='batchmean')/9
    expected += (ne.sigmoid()-edit.detach().sigmoid()).square().mean()
    loss = paired_consistency(cat, edit, nc, ne)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert cat.grad is None and edit.grad is None
    assert torch.isfinite(nc.grad).all() and torch.isfinite(ne.grad).all()


def test_identical_predictions_have_zero_consistency():
    cat, edit = torch.randn(2, 4, 3, 3), torch.randn(2, 3, 3)
    torch.testing.assert_close(paired_consistency(cat, edit, cat, edit), torch.tensor(0.), atol=1e-7, rtol=0)


def test_category_mixup_has_finite_gradients_and_ignores_source_grid():
    torch.manual_seed(3)
    cat = torch.randn(2, 4, 3, 3, requires_grad=True)
    y = {'target_grid': torch.randint(4, (2, 3, 3)),
         'edit_grid': torch.randint(2, (2, 3, 3)), 'source_grid': torch.zeros(2, 3, 3)}
    loss = category_mixup_loss(cat, y, .05)
    y['source_grid'].fill_(3)
    torch.testing.assert_close(loss, category_mixup_loss(cat, y, .05))
    loss.backward()
    assert torch.isfinite(cat.grad).all() and cat.grad.abs().sum() > 0


def test_empty_and_full_edit_masks_remain_finite():
    cat = torch.randn(2, 4, 3, 3)
    for value in (0, 1):
        y = {'target_grid': torch.zeros(2, 3, 3, dtype=torch.long),
             'edit_grid': torch.full((2, 3, 3), value)}
        assert torch.isfinite(category_mixup_loss(cat, y))
