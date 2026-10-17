"""CPU independent synthetic ridge normal equations; no optical/teacher model."""
import os
import numpy as np
import torch
import caltech20_same_linear_teacher_ridge as m
def main():
    out=m.CPU;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();m.claim(out.with_name(out.name+'.claim.json'),'synthetic_ridge_and_positive_scalar_calibration');torch.set_num_threads(4);r=np.random.default_rng(17);X=r.normal(size=(17,6));Y=r.normal(size=(17,3));Y-=Y.mean(1,keepdims=True);Y[2]=0;A=X.T@X;lam=.01*np.trace(A)/len(A);expected=np.linalg.solve(A+lam*np.eye(6),X.T@Y);actual=torch.linalg.solve(torch.tensor(X).T@torch.tensor(X)+lam*torch.eye(6,dtype=torch.float64),torch.tensor(X).T@torch.tensor(Y)).numpy();error=float(np.abs(actual-expected).max());assert error<1e-11
    z=r.normal(size=(40,20));y=np.tile(np.arange(20),2);a=m.temperature(z,y,[.05,5.],60);prob=m.probability(a*z);weights=len(y)/(20*np.bincount(y,minlength=20)[y]);derivative=float(np.mean(weights*((prob*z).sum(1)-z[np.arange(len(y)),y])));assert .05<=a<=5 and np.array_equal(z.argmax(1),(a*z).argmax(1))
    if .05+1e-8<a<5-1e-8:assert abs(derivative)<1e-12
    out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,ridge_torch_numpy_maxabs=error,positive_scale=a,balanced_NLL_scale_derivative=derivative,synthetic_linear_solves=1,optical_models=0,optical_forward_calls=0,teacher_backbone_forward_calls=0,real_images_read=0,TEST_arrays_read=0),exclusive=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        out=m.CPU;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
