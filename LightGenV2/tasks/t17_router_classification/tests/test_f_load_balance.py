import numpy as np
import pytest
from LightGenV2.tasks.t17_router_classification.report_f_load_balance import usage_statistics


def test_uniform_power_does_not_hide_hard_route_collapse():
    hard=np.tile([.5,.5,0,0],(8,1));dense=np.full((8,4),.25)
    result=usage_statistics(hard,dense)
    assert result['selection_slot_share']==[.5,.5,0.,0.]
    assert result['per_image_selection_frequency']==[1.,1.,0.,0.]
    assert result['unused_experts']==[3,4]
    assert result['normalized_selection_entropy']==pytest.approx(.5)


def test_actual_top2_uniform_load_definitions_and_pair_change():
    hard=np.array([[.5,.5,0,0],[0,0,.5,.5],[.5,0,.5,0],[0,.5,0,.5]])
    value=usage_statistics(hard,hard)
    assert value['selection_slot_share']==[.25]*4
    assert value['per_image_selection_frequency']==[.5]*4
    assert value['normalized_selection_entropy']==pytest.approx(1.)
    assert value['selection_share_CV']==0
    assert value['fraction_outside_most_common_pair']==.75


def test_dense_route_cannot_be_presented_as_actual_top2():
    with pytest.raises(ValueError):usage_statistics(np.full((4,4),.25),np.full((4,4),.25))
