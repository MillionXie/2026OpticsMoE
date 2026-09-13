import random
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import training_pairs, retrieval_loss, phase_delta, fitting_groups
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import coil_records, validate_rows


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


def test_multi_view_bank_masks_self_and_has_finite_gradients():
    z = torch.randn(4,64,requires_grad=True)
    bank = torch.randn(4,64)
    labels = torch.tensor([0,1,0,1])
    bank_labels = torch.tensor([0,0,1,1])
    excluded = torch.tensor([[True,False,False,False],[False,False,True,False]])
    loss,_ = retrieval_loss(z,labels,bank,2,bank_labels,excluded)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError,match='nonself'):
        retrieval_loss(z,labels,bank,2,bank_labels,torch.ones(2,4,dtype=torch.bool))


def test_clean_train_ranking_excludes_identical_photo():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import rank_instances
    rows = [dict(sample_id='a', product_id='x',split='gallery'),
            dict(sample_id='b', product_id='x',split='gallery'),
            dict(sample_id='c', product_id='y',split='gallery'),
            dict(sample_id='a', product_id='x',split='query')]
    z = torch.zeros(4,64); z[0,0]=z[3,0]=1; z[1,1]=1; z[2,0]=.9;z[2,1]=.1
    with pytest.raises(ValueError,match='Query image'):
        rank_instances(z,rows)
    metrics,predictions=rank_instances(z,rows,exclude_self=True)
    assert metrics['hit_at_1']==0 and predictions[0]['top1_sample_id']=='c'


def test_fresh_initialization_never_copies_old_trainable_state(tmp_path):
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import load_initial_weights
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = torch.nn.Linear(1,1)
            for name in ('vision','language'):
                m=torch.nn.Module();m.alpha_bounds=(.401,.95)
                m.block1_optical_fusion_logit=torch.nn.Parameter(torch.zeros(()))
                m.block2_optical_fusion_logit=torch.nn.Parameter(torch.zeros(()))
                setattr(self,name,m)
            self.readout=torch.nn.Linear(1,1)
    m=Tiny();original=m.readout.weight.detach().clone()
    torch.save({'state_dict': {'frontend.weight': torch.ones(1,1)*.123,
                              'frontend.bias': torch.zeros(1)}},tmp_path/'best.pt')
    poisoned={'state_dict': {n: torch.full_like(p,999) for n,p in m.state_dict().items()}}
    load_initial_weights(m,poisoned,tmp_path,True)
    assert torch.equal(m.readout.weight,original)
    assert torch.allclose(m.frontend.weight,torch.tensor([[.123]]))
    assert abs(float(.401+(.95-.401)*m.vision.block1_optical_fusion_logit.sigmoid())-.45)<1e-6


def test_phase_report_includes_router_experts_global_and_is_circular():
    class Tiny(torch.nn.Module):
        def named_parameters(self):
            yield 'vision.optics.raw_router_phase', torch.nn.Parameter(torch.tensor(0.))
            yield 'vision.optics.experts.0', torch.nn.Parameter(torch.tensor(0.))
            yield 'vision.optics.global_phase', torch.nn.Parameter(torch.tensor(0.))
    initial = {n: torch.tensor(torch.pi + 2 * torch.pi) for n, _ in Tiny().named_parameters()}
    assert all(v < 1e-6 for v in phase_delta(Tiny(), initial).values())


def test_coil_fitting_never_reads_heldout_gallery_or_objects():
    g = validate_rows(coil_records())
    fit = fitting_groups(dict(protocol='heldout40_objects_four_gallery_views_eight_queries_v1'), g)
    assert len(fit['train']) == 4080 and len(fit['gallery']) == 240
    ids = {r['sample_id'] for r in fit['train'] + fit['gallery']}
    assert ids == {r['sample_id'] for r in g['train']}
    assert not ids & {r['sample_id'] for r in g['gallery'] + g['query']}
    assert not {r['sample_id'] for r in fit['train']} & {r['sample_id'] for r in fit['gallery']}
    assert all(r['source_split'] == 'train' for r in fit['gallery'])
    for seed in range(3):
        pair, label = training_pairs(fit, random.Random(seed), 8)
        assert label[:8].tolist() == label[8:].tolist()
        assert all(r['sample_id'] in ids for r in pair)
    g['train'][0]['product_id'] = g['query'][0]['product_id']
    with pytest.raises(ValueError, match='leaked'):
        fitting_groups(dict(protocol='heldout40_objects_four_gallery_views_eight_queries_v1'), g)


def test_multi_reference_loss_uses_all_positives_not_only_first():
    z = torch.zeros(4, 64)
    z[0, 0] = z[2, 0] = 1
    z[1, 1] = z[3, 1] = 1
    labels = torch.tensor([0, 1, 0, 1])
    bank = torch.zeros(4, 64)
    bank[0, 2] = bank[1, 0] = bank[2, 3] = bank[3, 1] = 1
    loss, hit = retrieval_loss(z, labels, bank, 2, torch.tensor([0, 0, 1, 1]))
    assert hit == 1 and loss < .01
    with pytest.raises(ValueError, match='without'):
        retrieval_loss(z, labels, bank, 2, torch.tensor([0, 0, 0, 0]))


def test_removed_optics_never_reports_stale_router_cache(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_adapt as m
    class Removed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = self.language = SimpleNamespace(remove_optical=True)
            self.metadata = {'input_preprocessing': 'contain_white'}
        def forward(self, batch):
            return torch.ones(len(batch), 64)
    monkeypatch.setattr(m, 'picture', lambda *a: None)
    monkeypatch.setattr(m, 'inputs', lambda p, images, d: images)
    z, audit = m.encode_rows(Removed(), None, [dict(image_path='a')] * 3, tmp_path, torch.device('cpu'), 2, routing=True)
    assert z.shape == (3, 64)
    assert audit['executed'] is False and 'vision' not in audit


def test_assessment_preserves_manifest_gallery_tie_order(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_adapt as m
    g = dict(gallery=[dict(sample_id='gb', product_id='b', split='gallery'),
                      dict(sample_id='ga', product_id='a', split='gallery')],
             query=[dict(sample_id='qb', product_id='b', split='query')],
             train=[dict(sample_id='tb', product_id='b', split='train')])
    monkeypatch.setattr(m, 'encode_rows', lambda model, processor, rows, *a, **k: (torch.ones(len(rows), 64), {}))
    result = m.assessment(None, None, g, SimpleNamespace(data=tmp_path, batch_size=4), torch.device('cpu'))
    assert result['test']['hit_at_1'] == 1  # manifest b-first, not sorted a-first
