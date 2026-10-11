import importlib.util, sys, types
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
pytestmark=pytest.mark.skipif(sys.platform=='win32',reason='Verify native Torch on Linux; do not change shared Windows environment')


def test_detector_native_class_counts():
    try:
        import fast_optical as fast
    except (ImportError,OSError):
        pytest.skip('Native Torch unavailable')
    for count in [9,17,30,60]:
        d=fast.detector(count)
        assert sum(d['N_det_sets'])==count
        assert all(0<=x and x+n*32+(n-1)*28<=500 for x,n in zip(d['start_pos_x_per_row'],d['N_det_sets']))


def test_batched_expert_keeps_rho_and_gradients():
    try:
        import fast_optical as fast
    except (ImportError,OSError):
        pytest.skip('Native Torch unavailable')
    torch=fast.torch
    class Phase:
        def __init__(self,rho):
            self.value=torch.nn.Parameter(torch.linspace(-1,1,16).reshape(4,4))
            self.residual_rho=rho
        def get_phase(self):return self.value
        def _dropout_enabled(self):return False
    for rho in [0.,.3]:
        phases=[Phase(rho) for _ in range(9)]
        experts=[types.SimpleNamespace(phase_layers=[p],propagation_padding=0,size=4,
                     propagators=[types.SimpleNamespace(propagator=lambda x:torch.fft.ifft2(torch.fft.fft2(x)))]) for p in phases]
        bank=types.SimpleNamespace(experts=experts)
        fields=[torch.ones(2,4,4,dtype=torch.complex64)*(i+1) for i in range(9)]
        result=fast.expert_stage_batched(bank,0,fields)
        expected=[fields[i]*((1-rho)*torch.exp(1j*p.value)+rho) for i,p in enumerate(phases)]
        assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(result,expected))
        sum(x.real.sum() for x in result).backward()
        assert all(p.value.grad is not None and torch.isfinite(p.value.grad).all() and p.value.grad.norm()>0 for p in phases)
