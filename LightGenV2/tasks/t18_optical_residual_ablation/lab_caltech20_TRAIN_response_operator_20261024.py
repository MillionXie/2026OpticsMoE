"""Training/diagnostic E2 response only; checkpoint state remains native.

No class conversion, monkeypatch, device control or deployed module insertion.
Native inference always reloads MeasuredPrefix via the original model_for.
"""
import torch
from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix
from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import spatial_intensity


def validate_response(r):
    assert len(r['phase_curve_radians'])==17 and len(r['pupil_coefficients_radians'])==8
    curve=r['phase_curve_radians']
    assert curve[0]==0 and all(b>a for a,b in zip(curve,curve[1:]))
    assert .25*2*torch.pi<=curve[-1]<=2*2*torch.pi
    assert 0<=r['sigma_pixels']<=12 and 0<=r['relative_background']<=3
    assert all(abs(x)<=12*torch.pi for x in r['pupil_coefficients_radians'])


class TRAINResponsePrefix(MeasuredPrefix):
    """Same native state dictionary, E2 observation formula during optimization."""
    def __init__(self,state,response):
        super().__init__(4);validate_response(response)
        self.load_state_dict(state,strict=True)
        self._response={k:response[k] for k in ('phase_curve_radians','sigma_pixels','relative_background','pupil_coefficients_radians')}
        assert set(self.state_dict())==set(state)
        assert all(torch.equal(v,self.state_dict()[k]) for k,v in state.items())

    def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
        if stage!='2' or stage in self.measured or callback is not None:
            return super().capture(amplitude,stage,weights,callback,ids)
        _,amp,exact=self.physical_fields(amplitude,stage,weights)
        plane=self.plane(stage)
        turns=(torch.remainder(torch.angle(plane),2*torch.pi)/(2*torch.pi))[20:498,20:498]
        # Preserve original physical gray values while retaining the original STE
        # derivative through phase. No alternate inference quantizer or encoding.
        turns=turns+(exact-turns).detach()
        r=self._response;curve=amp.new_tensor(r['phase_curve_radians']);sigma=amp.new_tensor(r['sigma_pixels'])
        bg=amp.new_tensor(r['relative_background']);coeff=amp.new_tensor(r['pupil_coefficients_radians'])
        intensity=spatial_intensity(amp,turns,self.propagator,curve,sigma,bg,coeff)
        assert intensity.shape==(len(amplitude),478,478) and torch.isfinite(intensity).all()
        self.last[stage]=dict(amplitude=amp.detach(),phase_gray=(exact.detach()*256).to(torch.uint8),intensity=intensity.detach())
        return intensity


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    native=MeasuredPrefix(4).cpu().eval();state={k:v.detach().clone() for k,v in native.state_dict().items()}
    nominal=dict(phase_curve_radians=torch.linspace(0,2*torch.pi,17).tolist(),sigma_pixels=0.,relative_background=0.,pupil_coefficients_radians=[0.]*8)
    candidate=TRAINResponsePrefix(state,nominal).eval()
    amplitude=torch.rand(2,224,224,requires_grad=True);weights=torch.full((2,4),.5)
    with torch.no_grad():expected=native.capture(amplitude,'2',weights)
    result=candidate.capture(amplitude,'2',weights)
    assert torch.allclose(expected,result,atol=3e-6,rtol=3e-6),float((expected-result).abs().max())
    yy,xx=torch.meshgrid(torch.linspace(0,1,478),torch.linspace(0,1,478),indexing='ij')
    loss=(result*(xx+yy.square())).mean();loss.backward()
    phase_norms=[float(candidate.phase_sets[k].grad.norm()) for k in range(5,9)]
    assert all(x>0 for x in phase_norms) and amplitude.grad is not None and torch.isfinite(amplitude.grad).all()
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in state.items())
    # An actual recorded CCD must bypass the response formula exactly.
    actual=torch.randint(0,255,(2,478,478),dtype=torch.uint8);candidate.measured={'2':actual}
    assert torch.equal(candidate.capture(amplitude,'2',weights),actual.float())
    restored=MeasuredPrefix(4);restored.load_state_dict(candidate.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in state.items())
    print(dict(passed=True,nominal_maxabs=float((expected-result).abs().max()),original_state_keys_identical=True,original_model_reload_exact=True,
        E2_phase_gradient_norms=phase_norms,input_gradient_finite=True,measured_CCD_bypass_exact=True,SDK_used=False,new_inference_layers=0))


if __name__=='__main__':selftest()
