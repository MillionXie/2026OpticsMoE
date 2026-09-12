import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.readout_calibration import fit_subspace,fold_transform


def test_subspace_is_full_rank_and_leaves_class_differences_unchanged():
    torch.manual_seed(4); x=torch.randn(40,64); labels=torch.arange(4).repeat_interleave(10)
    w,report=fit_subspace(x,labels,.5)
    torch.testing.assert_close(w,w.T,atol=1e-12,rtol=0)
    eig=torch.linalg.eigvalsh(w)
    torch.testing.assert_close(eig[:61],torch.full((61,),.5,dtype=torch.float64))
    torch.testing.assert_close(eig[61:],torch.ones(3,dtype=torch.float64))
    means=torch.stack([F.normalize(x.double(),dim=-1)[labels==c].mean(0) for c in range(4)])
    differences=means-means.mean(0)
    torch.testing.assert_close(differences@w.T,differences)
    assert report['training_rows']==40 and report['subspace_rank']==3
    identity,_=fit_subspace(x,labels,1.)
    assert torch.equal(identity,torch.eye(64,dtype=torch.float64))


@pytest.mark.parametrize('retention',[True,0,-.1,1.1,float('nan')])
def test_subspace_rejects_degenerate_or_invalid_fit(retention):
    with pytest.raises(ValueError):fit_subspace(torch.ones(20,64),torch.arange(2).repeat_interleave(10),retention)
    with pytest.raises(ValueError):fit_subspace(torch.ones(20,64),torch.arange(2).repeat_interleave(10),.5)


def test_fold_commutes_with_final_normalization_and_preserves_all_other_weights():
    torch.manual_seed(7)
    x=torch.randn(40,64); labels=torch.arange(4).repeat_interleave(10)
    transform,_=fit_subspace(x,labels,.5)
    phase=torch.randn(5,5); weight=torch.randn(64,384);bias=torch.randn(64)
    payload=dict(metadata={'retrieval_head':'linear64'},state_dict={
        'readout.projection.weight':weight,'readout.projection.bias':bias,'vision.optics.global_phase':phase},
        auxiliary_training_head={'old':torch.ones(1)},selection_score=.9)
    fitted=fold_transform(payload,transform,{'method':'test'})
    assert fitted['state_dict']['vision.optics.global_phase'] is phase
    assert 'readout_calibration' not in payload['metadata']
    assert 'auxiliary_training_head' not in fitted and 'selection_score' not in fitted
    values=torch.randn(8,384)
    original=F.normalize(F.linear(values,weight,bias),dim=-1)
    expected=F.normalize(original.double()@transform.T,dim=-1)
    actual=F.normalize(F.linear(values,fitted['state_dict']['readout.projection.weight'],fitted['state_dict']['readout.projection.bias']),dim=-1)
    torch.testing.assert_close(actual.double(),expected,rtol=1e-4,atol=2e-7)
