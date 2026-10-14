"""New15-class CPU admission and native proofs; never uses test images."""
import argparse,inspect,os,subprocess,textwrap,types
from pathlib import Path
import numpy as np
import torch
import resisc15 as study
import check_resisc20_input as sealed
import export_resisc15 as export
# The unchanged BMP/routing replay is reused with exactly2 fifteen-output cases.
def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
    m=study.manifest(Path(study.plan()['parent_data']));cfg,function=study.bind()
    x,y,ids=study.load_fit(Path(cfg['data']),'train',m);assert len(y)==8398 and len(set(y.tolist()))==15
    vx,vy,vids=study.load_fit(Path(cfg['data']),'val',m);assert len(vy)==1049
    eye=torch.arange(15);q=study.s.engine.F_onehot(eye,.08);assert q.shape==(15,15) and torch.allclose(q.sum(1),torch.ones(15))
    labelmap=study.subset_labels(np.arange(20));assert np.array_equal(labelmap,np.arange(15))
    del x,y,vx,vy
    for rho in (.3,0.):
        torch.manual_seed(17);model=study.SubsetMoE(4,rho);assert sum(p.numel() for p in model.parameters())==908552 and len(model.detectors)==15
        assert torch.equal(model.detectors,study.s.r.CorrectedMoE(4,rho).detectors[list(study.INDICES)])
    import audit_resisc15 as audit; import types as _types
    try:audit.audit(_types.SimpleNamespace(base=study.HERE/'runs/smoke/wrong15run_never_created',out=out.with_name(out.name+'_badout'),data=Path(cfg['data'])))
    except AssertionError:pass
    else:raise AssertionError('Wrong run admission did not reject')
    try:study.load_fit(Path(cfg['data']),'test',m)
    except AssertionError:pass
    else:raise AssertionError('Test loading must reject')
    src=textwrap.dedent(inspect.getsource(sealed.check))
    changes={"s.r.CorrectedMoE(4,rho)":"study.SubsetMoE(4,rho)","range(20)":"range(15)","static_packages=3":"static_packages=2","model_forward_calls=18":"model_forward_calls=12","dataset_arrays_read=0":"dataset_arrays_read=2"}
    for old,new in changes.items():assert src.count(old)==1;src=src.replace(old,new)
    view=types.SimpleNamespace(CASES={k:(100,v) for k,v in study.CASES.items()},config=lambda case:study.config(),sources=study.sources,encode=study.encode,SubsetMoE=study.SubsetMoE)
    ns=dict(sealed.__dict__);ns.update(study=view,export=export);exec(compile(src,__file__+':sealed15CPU_native','exec'),ns);ns['check'](out)
    value=study.s.read(out/'checks.json');value.update(class_count=15,train_samples=8398,val_samples=1049,optimizer_steps_per_epoch=525,tail_batch=14,class_selection_fixed_random=True,role_and_ID_preserved=True,plan_sha256=study.s.sha(study.PLAN),subset_manifest_sha256=study.s.sha(study.private_manifest()),source_count=len(study.sources()),model_forward_calls=12,test_arrays_read=0,test_evaluations=0)
    study.s.save(out/'checks.json',value)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        if a.out.exists() and not (a.out/'failure.json').exists():study.s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
