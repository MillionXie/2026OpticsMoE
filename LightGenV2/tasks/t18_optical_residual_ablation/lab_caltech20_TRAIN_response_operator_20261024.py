"""Training/diagnostic E2 response only; checkpoint state remains native.

No class conversion, monkeypatch, device control or deployed module insertion.
Native inference always reloads MeasuredPrefix via the original model_for.
"""
import torch
from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix
from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import spatial_intensity
from lab_caltech20_TRAIN_flat_response_20261024 import apply_map


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


class TRAINPhotometricPrefix(MeasuredPrefix):
    """Training-phase E2 prediction only; never a deployed image correction."""
    def __init__(self,state,gain,background):
        super().__init__(4);self.load_state_dict(state,strict=True)
        assert tuple(gain.shape)==tuple(background.shape)==(8,8)
        assert bool(torch.isfinite(gain).all()) and bool(torch.isfinite(background).all())
        assert float(gain.min())>=.05 and float(gain.max())<=4 and float(background.min())>=0 and float(background.max())<=4
        self._gain=gain.tolist();self._background=background.tolist()
        assert set(self.state_dict())==set(state) and all(torch.equal(v,self.state_dict()[k]) for k,v in state.items())

    def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
        if stage!='2' or stage in self.measured or callback is not None:
            return super().capture(amplitude,stage,weights,callback,ids)
        field,amp,phase=self.physical_fields(amplitude,stage,weights)
        nominal=self.propagator(field).abs().square()[:,20:498,20:498]
        intensity=apply_map(nominal,nominal.new_tensor(self._gain),nominal.new_tensor(self._background))
        self.last[stage]=dict(amplitude=amp.detach(),phase_gray=(phase.detach()*256).to(torch.uint8),intensity=intensity.detach())
        return intensity


def photometric_selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    native=MeasuredPrefix(4).eval();state={k:v.detach().clone() for k,v in native.state_dict().items()}
    candidate=TRAINPhotometricPrefix(state,torch.ones(8,8),torch.zeros(8,8)).eval()
    x=torch.rand(2,224,224)*.25
    shared={k:torch.randint(0,255,(2,478,478),dtype=torch.uint8) for k in ('router','0','1')}
    native.measured=shared;candidate.measured=shared
    with torch.no_grad():expected=native(x)[0];actual=candidate(x)[0]
    assert torch.allclose(actual,expected,atol=3e-6,rtol=3e-6),float((actual-expected).abs().max())
    candidate.train();weights=torch.full((2,4),.5);amplitude=torch.rand(2,224,224,requires_grad=True)
    value=candidate.capture(amplitude,'2',weights);weight=torch.linspace(0,1,478)[None,None,:]
    (value*weight).mean().backward();norms=[float(candidate.phase_sets[i].grad.norm()) for i in range(5,9)]
    assert all(v>0 for v in norms) and amplitude.grad is not None and torch.isfinite(amplitude.grad).all()
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in state.items())
    native.load_state_dict(candidate.state_dict(),strict=True)
    real=torch.randint(0,255,(2,478,478),dtype=torch.uint8);candidate.measured={'2':real}
    assert torch.equal(candidate.capture(amplitude,'2',weights),real.float())
    print(dict(passed=True,nominal_original_probability_maxabs=float((actual-expected).abs().max()),same_state_keys=True,native_reload_exact=True,phase_gradient_norms=norms,
        input_gradient_finite=True,real_CCD_bypass_exact=True,SDK_used=False,GPU_used=False,new_inference_layers=0))


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
