import ast
from pathlib import Path
import numpy as np

ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_native_response_fit_20261024.py'


def test_finite_TRAIN_only_pair_training_and_native_admission():
    s=ENTRY.read_text();ast.parse(s)
    assert 'fitrows=[v for v in pack' in s and "v['split']=='fit'" in s and 'len(fitrows)==len(held)==80' in s
    assert 'chunk=[fitrows[i] for i in order[begin:begin+4]]' in s
    for bad in ("data['test']",'TEST_labels','Bench(','capture.acquire(','torch.save(','saved_model(','grid_sample'):assert bad not in s
    assert "final[tag]['native_optical_PCC']>=baseline[tag]['native_optical_PCC']+.03" in s
    assert "final[tag]['coarse16_PCC']>=baseline[tag]['coarse16_PCC']-.01" in s
    assert 'range(1,11)' in s and 'time.monotonic()-fitstart>=300' in s


def test_native_loss_not_merely_coarse_image_and_no_inference_update():
    s=ENTRY.read_text()
    assert "model.nonlinear(predicted,'global')" in s and 'return native+.25*pixel+.25*coarse' in s
    assert 'kernel=torch.exp(-2*torch.pi**2*sigma.square()' in s
    assert 'classification_accuracy_updated=False' in s and 'new_inference_layers=0' in s
    assert 'model_checkpoint_written=False' in s


def test_contrast_zero_and_affine_invariant_numpy_equivalence():
    class A(np.ndarray):
        def square(self):return np.square(self).view(A)
        def mean(self,axes=None,keepdim=False):return np.mean(np.asarray(self),axis=axes,keepdims=keepdim).view(A)
        def sqrt(self):return np.sqrt(self).view(A)
        def clamp_min(self,x):return np.maximum(self,x).view(A)
    node=next(v for v in ast.parse(ENTRY.read_text()).body if isinstance(v,ast.FunctionDef) and v.name=='contrast_loss');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns)
    x=np.random.default_rng(73).normal(size=(4,16,16)).view(A);f=ns['contrast_loss']
    assert abs(float(f(x,x)))<1e-12 and abs(float(f(3*x+2,x)))<1e-12
