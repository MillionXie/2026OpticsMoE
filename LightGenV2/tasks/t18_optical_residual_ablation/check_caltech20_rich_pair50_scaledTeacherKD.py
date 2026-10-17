"""Own calibration algebra and frozen-parent admission; no old proof rerun."""
import ast,inspect,argparse,os
from pathlib import Path
import numpy as np,torch
import check_caltech20_rich_pair50_balancedKD as frozen
import caltech20_rich_pair50_scaledTeacherKD as m
SYNTHETIC=m.HERE/'runs/smoke/caltech20_rich_pair50_scaledTeacherKD_synthetic_cpu_20261017'
for node in ast.parse(inspect.getsource(frozen)).body:
    if isinstance(node,ast.FunctionDef):exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())
original_synthetic_update=synthetic_update
def synthetic_update(cfg):
    result=original_synthetic_update(cfg)
    teacher=np.array([[12.,-3.,2.],[-4.,8.,1.]],np.float64)
    qt=np.exp(teacher/16-(teacher/16).max(1,keepdims=True));qt/=qt.sum(1,keepdims=True)
    qwrong=np.exp(teacher/4-(teacher/4).max(1,keepdims=True));qwrong/=qwrong.sum(1,keepdims=True)
    assert abs(qt-qwrong).max()>.1
    student=torch.tensor([[.4,-.2,.1],[-.3,.5,.2]],dtype=torch.float64,requires_grad=True)
    T=cfg['temperature'];q=torch.tensor(qt);logs=torch.log_softmax(student/T,1);loss=T*T*(q*(q.log()-logs)).sum(1).mean();gradient=torch.autograd.grad(loss,student)[0].numpy()
    expected=T*(torch.softmax(student.detach()/T,1).numpy()-qt)/2
    error=float(np.abs(gradient-expected).max());assert error<1e-12 and cfg['teacher_temperature']==16
    result.update(teacher_temperature=16.,student_temperature=4.,independent_calibrated_target_gradient_maxabs=error,wrong_teacher_T4_rejected=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    try:check_synthetic(a.out) if a.synthetic else check(a.out)
    except BaseException as e:
        fail=a.out.with_name(a.out.name+'.failure.json')
        if not fail.exists():m.s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
