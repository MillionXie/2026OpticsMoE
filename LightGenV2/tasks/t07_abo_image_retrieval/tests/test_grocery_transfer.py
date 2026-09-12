import random
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.grocery_transfer import training_pairs, retrieval_loss, phase_delta


def groups():
    return {s: [dict(sample_id=f'{s}{i}', product_id=str(i), split=s) for i in range(4)]
            for s in ('train', 'gallery', 'query')}


def test_sampler_uses_only_train_and_public_iconic_not_test():
    g = groups()
    a, labels = training_pairs(g, random.Random(42), 3)
    assert [r['split'] for r in a] == ['train'] * 3 + ['gallery'] * 3
    assert labels[:3].tolist() == labels[3:].tolist()
    assert len(set(labels.tolist())) == 3
    assert a == training_pairs(g, random.Random(42), 3)[0]
    g['train'][0]['split'] = 'query'
    with pytest.raises(ValueError, match='Non-TRAIN'):
        training_pairs(g, random.Random(42), 3)


def test_training_loss_updates_both_domains_but_detaches_gallery_bank():
    torch.manual_seed(42)
    z = torch.randn(8, 64, requires_grad=True)
    bank = torch.randn(81, 64, requires_grad=True)
    labels = torch.tensor([0, 20, 50, 80] * 2)
    loss, hit = retrieval_loss(z, labels, bank, 4)
    assert torch.isfinite(loss) and 0 <= hit <= 1
    loss.backward()
    assert z.grad[:4].norm() > 0 and z.grad[4:].norm() > 0
    assert bank.grad is None
    with pytest.raises(ValueError, match='paired'):
        retrieval_loss(z, labels, bank, 3)


def test_phase_report_includes_router_experts_global_and_is_circular():
    class Tiny(torch.nn.Module):
        def named_parameters(self):
            yield 'vision.optics.raw_router_phase', torch.nn.Parameter(torch.tensor(0.))
            yield 'vision.optics.experts.0', torch.nn.Parameter(torch.tensor(0.))
            yield 'vision.optics.global_phase', torch.nn.Parameter(torch.tensor(0.))
    initial = {n: torch.tensor(torch.pi + 2 * torch.pi) for n, _ in Tiny().named_parameters()}
    assert all(v < 1e-6 for v in phase_delta(Tiny(), initial).values())
