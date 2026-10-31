import copy
import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG,load_historical_F_phases


def test_historical_three_phase_initialization_preserves_values_without_importing_old_function():
    model=LinearReadoutLG(2,10);head=model.shared_head.weight.detach().clone()
    payload=dict(model={n:torch.randn_like(v) for n,v in model.state_dict().items() if n!='shared_head.weight'},optical_config=dict(model.cfg,oeo_activation='none'))
    info=load_historical_F_phases(model,payload)
    assert info['all_imported_phases_bitwise_equal'] and info['historical_oeo']=='none'
    assert model.cfg['oeo_activation']=='intensity_softsign' and torch.equal(head,model.shared_head.weight)
    assert sum(isinstance(m,torch.nn.Linear) for m in model.modules())==1
    assert all(torch.equal(model.state_dict()[n],v) for n,v in payload['model'].items())
    bad=copy.deepcopy(payload);bad['optical_config']['propagation_m']=.2
    with pytest.raises(ValueError,match='geometry'):load_historical_F_phases(model,bad)


def test_historical_import_cannot_silently_add_CNN_or_mismatch_depth():
    model=LinearReadoutLG(2,10);payload=dict(model={n:v for n,v in model.state_dict().items() if n!='shared_head.weight'},optical_config=model.cfg)
    payload['model']['CNN.weight']=torch.ones(1)
    with pytest.raises(ValueError,match='three'):load_historical_F_phases(model,payload)
    with pytest.raises(ValueError,match='two-layer'):load_historical_F_phases(LinearReadoutLG(4,10),payload)
