import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout import within_sku_whiten_projection


def fixture():
    generator=torch.Generator().manual_seed(164)
    return (torch.randn(80,64,generator=generator),torch.arange(10).repeat_interleave(8),
            torch.randn(64,384,generator=generator),torch.randn(64,generator=generator))


def test_whitening_zero_exact_and_label_renumbering_invariant():
    z,y,w,b=fixture()
    a,c,_=within_sku_whiten_projection(z,y,w,b,0.)
    assert torch.equal(a,w) and torch.equal(c,b)
    a,c,audit=within_sku_whiten_projection(z,y,w,b,.1)
    d,e,_=within_sku_whiten_projection(z,y*10+7,w,b,.1)
    assert torch.equal(a,d) and torch.equal(c,e)
    assert not audit['fitted_to_query'] and audit['training_rows']==80


def test_folded_head_equals_fixed_train_metric_no_new_layer():
    z,y,w,b=fixture()
    nw,nb,audit=within_sku_whiten_projection(z,y,w,b,.1)
    unit=F.normalize(z.double(),dim=1)
    centered=unit-unit.reshape(10,8,64).mean(1).repeat_interleave(8,0)
    cov=centered.T@centered/70;cov=cov*64/cov.trace()
    ev,u=torch.linalg.eigh(cov)
    matrix=(u*(.9+.1*ev).rsqrt()[None])@u.T
    x=torch.randn(5,384)
    expected=F.linear(x.double(),w.double(),b.double())@matrix.T
    assert torch.allclose(F.linear(x,nw,nb).double(),expected,atol=2e-5,rtol=1e-5)
    assert 0<audit['metric_gain_range'][0]<=audit['metric_gain_range'][1]<1.055


def test_whitening_rejects_missing_positive_and_degenerate_vectors():
    z,y,w,b=fixture()
    for strength in [-.1,1.,float('nan')]:
        with pytest.raises(ValueError):
            within_sku_whiten_projection(z,y,w,b,strength)
    with pytest.raises(ValueError):
        within_sku_whiten_projection(z,torch.arange(80),w,b,.1)
    with pytest.raises(ValueError):
        within_sku_whiten_projection(torch.ones_like(z),y,w,b,.1)
    with pytest.raises(ValueError):
        within_sku_whiten_projection(z.requires_grad_(),y,w,b,.1)


def test_closed_form_epoch_zero_is_reported_as_fitted_not_untrained():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import checkpoint_history
    payload=dict(manifest_sha256='abo',epoch=0,variant='metric',stage='readout_metric_fit',test_selected=True,
        readout_calibration=dict(method='within_sku_covariance',fitted_on_training_data=True,
                                 training_rows=1600,strength=.02))
    report=checkpoint_history(payload,'abo')
    assert report['fitted_on_this_dataset'] and report['test_selected']
    assert 'TRAIN-statistic' in report['checkpoint_origin']
    assert not checkpoint_history(payload,'other')['fitted_on_this_dataset']
    payload['readout_calibration']['strength']=0.
    assert not checkpoint_history(payload,'abo')['fitted_on_this_dataset']
