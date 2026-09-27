import math
import torch
from LightGenV2.tasks.t12_text_to_image.lab_shs8um.bounded_amplitude import encode
from LightGenV2.tasks.t12_text_to_image.lab_shs8um.robust_channel import coherent_modulation, RobustChannel


def test_bounded_phase_and_zero_preserved():
    field=torch.tensor([0j, .3+.4j, 2j],dtype=torch.complex64)
    output=encode(field,dict(kind='tanh',scale=.5))
    assert output[0]==0
    torch.testing.assert_close(output.abs(),torch.tanh(field.abs()/.5))
    torch.testing.assert_close(torch.angle(output[1:]),torch.angle(field[1:]))
    assert ((output.abs()*255).round()/255-output.abs()).abs().max()<=.5/255+1e-7


def test_power_not_amplitude_semantics():
    modulation=torch.ones(1,1,1,dtype=torch.complex64)
    eta=torch.tensor([[[.3]]])
    angle=torch.tensor([[[math.pi/2]]])
    mixed=coherent_modulation(modulation,eta,angle)
    torch.testing.assert_close(mixed.real,torch.tensor([[[math.sqrt(.7)]]]))
    torch.testing.assert_close(mixed.imag,torch.tensor([[[math.sqrt(.3)]]]))
    torch.testing.assert_close(mixed.abs().square(),torch.ones(1,1,1))
    assert (mixed*torch.zeros_like(mixed)).abs().sum()==0


def test_coherent_interference_is_retained():
    modulation=torch.ones(1,1,1,dtype=torch.complex64)
    eta=torch.tensor([[[.3]]])
    constructive=coherent_modulation(modulation,eta,torch.zeros(1,1,1)).abs().square()
    destructive=coherent_modulation(modulation,eta,torch.full((1,1,1),math.pi)).abs().square()
    assert constructive.item()>1.9 and destructive.item()<.1


def test_inactive_channel_camera_identity():
    channel=RobustChannel.__new__(RobustChannel);channel.active=False
    original=torch.rand(2,8,8)
    assert channel.camera('router',None,None,original) is original
