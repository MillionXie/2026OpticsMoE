import copy
import torch
import json
import pytest
from LightGenV2.tasks.t17_router_classification.train_regularized import augment_d4,update_ema
from LightGenV2.tasks.t17_router_classification.train_regularized import class_weights
from LightGenV2.tasks.t17_router_classification.evaluate_selected import reuse_parent_receipt
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256

def test_d4_preserves_rgb_alignment_power_and_empty_quadrant():
    original=torch.arange(112*112).reshape(112,112).float()
    x=torch.zeros(8,224,224)
    x[:,:112,:112]=original;x[:,:112,112:]=original*2;x[:,112:,:112]=original*3
    out=augment_d4(x,torch.arange(8))
    assert torch.count_nonzero(out[:,112:,112:])==0
    assert torch.allclose(out[:,:112,112:],out[:,:112,:112]*2)
    assert torch.allclose(out[:,112:,:112],out[:,:112,:112]*3)
    assert torch.allclose(out.square().sum((1,2)),x.square().sum((1,2)))
    assert torch.equal(out[0],x[0])
    assert not torch.equal(out[1],x[1])

def test_ema_blends_parameters_and_copies_buffers_without_changing_graph():
    model=torch.nn.BatchNorm1d(3);ema=copy.deepcopy(model)
    with torch.no_grad():model.weight.fill_(3);model.running_mean.fill_(5)
    update_ema(ema,model,.5)
    assert torch.equal(ema.weight,torch.full((3,),2.))
    assert torch.equal(ema.running_mean,model.running_mean)
    assert list(ema.state_dict())==list(model.state_dict())

def test_class_weights_depend_only_on_training_counts():
    y=torch.cat([torch.full((i+1,),i) for i in range(9)])
    assert torch.equal(class_weights(y,0),torch.ones(9))
    weights=class_weights(y,.5)
    assert torch.all(weights[:-1]>weights[1:])
    assert torch.allclose(weights.mean(),torch.tensor(1.))
    with pytest.raises(ValueError):class_weights(torch.zeros(5,dtype=torch.long),.5)

@pytest.mark.parametrize('mismatch',[False,True])
def test_parent_receipt_reuse_requires_elementwise_equal_weights(tmp_path,mismatch):
    parent=tmp_path/'parent';run=tmp_path/'candidate';parent.mkdir();run.mkdir()
    state={'w':torch.ones(2)}
    torch.save(dict(model=state),parent/'best_checkpoint.pt')
    digest=sha256(parent/'best_checkpoint.pt')
    (parent/'result.json').write_text(json.dumps(dict(accuracy=.8,checkpoint_sha256=digest)))
    (parent/'test_predictions.npz').write_bytes(b'existing receipt')
    ck=dict(weights_kind='parent',model={'w':torch.zeros(2) if mismatch else torch.ones(2)},
        config=dict(init_run=str(parent),initial_checkpoint=dict(sha256=digest)))
    torch.save(ck,run/'best_checkpoint.pt')
    if mismatch:
        with pytest.raises(ValueError,match='not identical'):reuse_parent_receipt(run,ck)
        assert not (run/'result.json').exists()
    else:
        assert reuse_parent_receipt(run,ck)
        assert json.loads((run/'result.json').read_text())['test_evaluations']==0
        assert (run/'test_predictions.npz').read_bytes()==b'existing receipt'
