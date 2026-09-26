import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.robust_training import prepare, shift_zero, attach
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import alpha_value


def test_alpha_conversion_preserves_actual_weight_and_source():
    key='vision.block1_optical_fusion_logit';raw=torch.tensor(-2.)
    source=dict(metadata=dict(fusion_alpha_min=.4001,fusion_alpha_max=.8,optical_training_noise={}),state_dict={key:raw})
    converted=prepare(source,dict(robust_alpha_min=.35,pixel_shift=1))
    assert torch.allclose(alpha_value(raw,(.4001,.8)),alpha_value(converted['state_dict'][key],(.35,.8)),atol=1e-7)
    assert source['metadata']['fusion_alpha_min']==.4001


def test_shift_gradient_and_zero_fill():
    torch.manual_seed(7)
    x=torch.ones(8,9,9,requires_grad=True);y=shift_zero(x)
    assert y.shape==x.shape and (y==0).any()
    y.sum().backward();assert torch.isfinite(x.grad).all()


def test_no_shift_is_noop():
    class Dummy:pass
    attach(Dummy(),dict(pixel_shift=0))
