import torch
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification

def test_sixteen_class_heads_match_and_optical_contract_stays_fixed():
    models=[FourRouterClassification(v,num_classes=16) for v in ('optical','electronic','d2nn')]
    assert all(m.shared_head.weight.shape==(16,784) for m in models)
    assert all(torch.equal(models[0].shared_head.weight,m.shared_head.weight) for m in models)
    assert all(m.global_phase.shape==(478,478) and m.width==518 for m in models)
    assert models[0].first_phase.shape==(4,224,224)
    assert models[1].electronic_router[-1].out_features==4
