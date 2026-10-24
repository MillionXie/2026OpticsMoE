import pytest
import torch
from LightGenV2.tasks.t19_resnet20_baselines.abo_protocol_diagnosis import catalog_identity,descriptor_margins


def test_closed_catalog_counts_and_multiple_positive_chance():
    x=catalog_identity(['a']*4+['b']*4,['a']*2+['b']*2,['a','b'])
    assert x['TEST_unseen_SKUs']==0 and x['shared_SKUs']==2
    assert x['TRAIN_views_per_SKU']=={4:2} and x['TEST_views_per_SKU']=={2:2}
    assert x['image_to_text_random_H1']==x['text_to_image_random_H1']==.5
    assert catalog_identity(['a'],['b'])['fraction_TEST_queries_of_seen_SKUs']==0


def test_TRAIN_leave_one_out_excludes_identical_index_not_other_same_SKU_views():
    x=torch.tensor([[1.,0.],[0.,1.],[.1,.9],[.9,.1]])
    skus=['a','a','b','b']
    normal=descriptor_margins(x,x,skus,skus)
    loo=descriptor_margins(x,x,skus,skus,exclude_self=True)
    assert normal['fraction_positive_margin']==1 and loo['fraction_positive_margin']==0
    assert loo['queries']==4 and loo['all_queries_retained']


def test_margin_diagnosis_refuses_missing_positive_or_wrong_self_order():
    x=torch.eye(2)
    with pytest.raises(ValueError):descriptor_margins(x,x,['a','b'],['b','a'],exclude_self=True)
    with pytest.raises(ValueError):descriptor_margins(x,x,['a','b'],['c','d'])
