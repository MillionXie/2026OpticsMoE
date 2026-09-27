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


def test_bmp_does_not_peak_normalize(tmp_path):
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.bounded_amplitude import save_bmps
    value=np.array([[[0.,.1],[.05,0.]]],dtype=np.float32)
    paths,scale=save_bmps(value,tmp_path,'router',['example'],SimpleNamespace(active_to_native=lambda x:x))
    pixels=np.asarray(Image.open(paths[0]))
    assert scale==1.0 and pixels[0,0]==0 and int(pixels.max())==26


def test_new_alpha_floor_preserves_current_value():
    from torch import nn
    from LightGenV2.tasks.t12_text_to_image.modeling import ScaleMatchedFusion
    from LightGenV2.tasks.t12_text_to_image.audited_unified import configure_fusion_bounds
    model=nn.Sequential(ScaleMatchedFusion(.49,.4,.75,1e-6))
    old=model[0].alpha.detach().clone()
    configure_fusion_bounds(model,.35)
    torch.testing.assert_close(old,model[0].alpha)
    assert model[0].minimum==.35


def test_spatial_camera_finite_and_nonnegative():
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.robust_channel import PROFILES
    channel=RobustChannel.__new__(RobustChannel)
    channel.active=True;channel.strength=1.;channel.profile=PROFILES['extreme']
    result=channel.camera('router',None,None,torch.zeros(2,16,16))
    assert result.isfinite().all() and (result>=0).all()


def test_physical_circular_tv_ignores_two_pi_wrap():
    from torch import nn
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.train_channel_robust import circular_phase_tv
    class Plane(nn.Module):
        def __init__(self):
            super().__init__();self.raw_phase=nn.Parameter(torch.tensor([[0.,2*math.pi],[0.,2*math.pi]]))
        def phase(self):return self.raw_phase
    assert circular_phase_tv(nn.Sequential(Plane())).item()<1e-6
