import pytest
import torch
from LightGenV2.tasks.t17_router_classification.evaluate_readout_checkpoint import assert_matching_optical_body


def test_existing_head_may_differ_but_no_optical_phase_may_change():
    parent={'phase':torch.ones(3,3),'shared_head.weight':torch.ones(2,4)};other={k:v.clone() for k,v in parent.items()};other['shared_head.weight'].zero_();assert_matching_optical_body(other,parent)
    other['phase'][0,0]=0
    with pytest.raises(ValueError,match='optical'):assert_matching_optical_body(other,parent)


def test_extra_or_missing_parameters_cannot_be_hidden_in_last_evaluation():
    with pytest.raises(ValueError,match='key set'):assert_matching_optical_body({'phase':torch.ones(3,3)},{'phase':torch.ones(3,3),'shared_head.weight':torch.ones(2,4)})
