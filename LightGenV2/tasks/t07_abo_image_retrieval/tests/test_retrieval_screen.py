from collections import Counter
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import coil_records, validate_rows, rank_instances, grocery_records


def test_trained_checkpoint_history_is_not_mislabeled_as_untrained_transfer():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import checkpoint_history
    payload = dict(manifest_sha256='coil', epoch=10, variant='live', test_selected=True)
    assert checkpoint_history(payload, 'coil')['fitted_on_this_dataset']
    assert checkpoint_history(payload, 'coil')['test_selected']
    assert not checkpoint_history(payload, 'grocery')['fitted_on_this_dataset']
    fallback = checkpoint_history(dict(payload, epoch=0, variant='initial'), 'coil')
    assert not fallback['fitted_on_this_dataset'] and fallback['test_selected']


def test_query_routing_collapse_cannot_be_hidden_by_gallery():
    from itertools import combinations
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import split_routing_report
    gallery=torch.zeros(12,4)
    for i,pair in enumerate(list(combinations(range(4),2))*2):gallery[i,list(pair)]=1
    query=torch.tensor([[1.,1.,0.,0.]]*6)
    masks=torch.cat([gallery,query]);original=masks.clone();rng=torch.random.get_rng_state()
    rows=[dict(split='gallery')]*12+[dict(split='query')]*6
    result=split_routing_report(dict(vision=masks,language=masks),rows)
    assert result['splits']['gallery']['eligible']
    assert not result['splits']['query']['eligible']
    assert result['splits']['query']['modalities']['vision']['selection_share']==[.5,.5,0.,0.]
    assert torch.equal(masks,original) and torch.equal(rng,torch.random.get_rng_state())
    balanced=torch.cat([gallery,gallery])
    result=split_routing_report(dict(vision=balanced,language=balanced),[dict(split='gallery')]*12+[dict(split='query')]*12)
    assert result['splits']['query']['eligible'] and result['splits']['gallery']['eligible']


def test_removed_routing_never_reports_stale_masks():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import split_routing_report
    assert not split_routing_report(None,[],removed=True)['executed']
    with pytest.raises(ValueError):split_routing_report({},[],removed=True)


@pytest.mark.parametrize('bad',['shape','soft','top1'])
def test_routing_masks_require_exact_top2_and_row_alignment(bad):
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import split_routing_report
    x=torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.]])
    if bad=='shape':x=x[:1]
    elif bad=='soft':x[0]=.5
    else:x[0,1]=0
    with pytest.raises(ValueError):split_routing_report(dict(vision=x,language=x),[dict(split='gallery'),dict(split='query')])


def test_coil_fixed_complete_split_and_angular_gap():
    rows = coil_records()
    assert rows == coil_records()
    assert Counter(r['split'] for r in rows) == dict(train=4320, gallery=160, query=320, unused_test_view=2400)
    g = validate_rows(rows)
    assert len({r['product_id'] for r in g['train']}) == 60
    assert len({r['product_id'] for r in g['query']}) == 40
    for query in g['query']:
        angles = [r['angle'] for r in g['gallery'] if r['product_id'] == query['product_id']]
        assert min(min(abs(query['angle']-a), 360-abs(query['angle']-a)) for a in angles) == 30


def test_identity_leakage_is_rejected():
    rows = coil_records()
    train = next(r for r in rows if r['split'] == 'train')
    query = next(r for r in rows if r['split'] == 'query')
    query['product_id'] = train['product_id']
    with pytest.raises(ValueError, match='leakage'):
        validate_rows(rows)


def test_ranking_uses_identity_not_nearby_category_or_query_itself():
    rows = [dict(sample_id='ga', product_id='a', split='gallery'),
            dict(sample_id='gb', product_id='b', split='gallery'),
            dict(sample_id='qa', product_id='a', split='query')]
    z = torch.zeros(3, 64)
    z[0, 0] = 1; z[1:, 1] = 1
    metrics, predictions = rank_instances(z, rows)
    assert metrics['hit_at_1'] == 0
    assert metrics['hit_at_5'] == 1
    assert predictions[0]['top1_sample_id'] == 'gb'
    z[2] = z[0]
    assert rank_instances(z, rows)[0]['hit_at_1'] == 1


@pytest.mark.parametrize('bad', ['zero', 'nan', 'shape'])
def test_invalid_vectors_fail(bad):
    rows = [dict(sample_id='g', product_id='a', split='gallery'), dict(sample_id='q', product_id='a', split='query')]
    z = torch.ones(2, 64)
    if bad == 'zero': z.zero_()
    elif bad == 'nan': z[0, 0] = float('nan')
    else: z = z[:, :32]
    with pytest.raises(ValueError): rank_instances(z, rows)


def test_category_protocol_does_not_claim_unseen_products():
    rows = [dict(sample_id=s, product_id='fine_class:0', split=s, image_path=s+'.jpg') for s in ['train','query','gallery']]
    with pytest.raises(ValueError): validate_rows(rows)
    assert len(validate_rows(rows, disjoint_products=False)['query']) == 1
    rows[1]['image_path'] = rows[0]['image_path']
    with pytest.raises(ValueError, match='Duplicate image path'): validate_rows(rows, disjoint_products=False)


def test_grocery_keeps_all_official_rows_without_subset_selection(tmp_path, monkeypatch):
    import csv
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_screen as module
    with (tmp_path/'classes.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['Class ID (int)','Coarse Class ID (int)','Iconic Image Path (str)'])
        writer.writerows((i,i//2,f'/iconic/{i}.jpg') for i in range(81))
    for split,count in [('train',2640),('test',2485),('val',296)]:
        with (tmp_path/f'{split}.txt').open('w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerows((f'{split}/{i}.jpg',i%81,(i%81)//2) for i in range(count))
    monkeypatch.setattr(module,'sha256',lambda p:str(p))
    rows=grocery_records(tmp_path)
    assert Counter(r['split'] for r in rows)==dict(train=2640,query=2485,gallery=81,unused_validation=296)
    assert {r['fine_class_id'] for r in rows if r['split']=='gallery'}==set(range(81))
    monkeypatch.setattr(module,'sha256',lambda p:'same-bytes')
    with pytest.raises(ValueError,match='Exact image duplication'): grocery_records(tmp_path)
