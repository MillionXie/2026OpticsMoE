"""Logging engineering repair only; preserve old failure and mathematical proof."""
import argparse,ast,builtins,contextlib,io,json,os
from pathlib import Path
import caltech20_equal60_logrepair as m
def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and m.slots()['total']==0
    old=m.s.read(m.HERE/'runs/smoke/caltech20_equal60_cpu_20261017/checks.json');assert old['passed'] and old['sources']==m.original.sources();m.engine.verify_sources(old['sources'],old['git_commit'])
    root=m.HERE/'runs/simulation/caltech20_rich20_current246_s17_equal60_20261017';state=m.s.read(root/'status.json');assert state['state']=='fit_failed' and not state['active'] and not state['pending'] and not (root/'once_tests').exists()
    failures={};oldgate={}
    for key in ('L6_rho03','L6_rho0','L4_rho03','L4_rho0'):
        f=m.s.read(root/key/'failure.json');assert f['error']=="NameError(\"name 'json' is not defined\")" or "name 'json' is not defined" in f['error'];failures[key]=m.s.sha(root/key/'failure.json');g=m.s.read(root/key/'numeric_gate.json');assert g['passed'];oldgate[key]=m.s.sha(root/key/'numeric_gate.json');ex=m.s.read(root/(key+'.fit_exit.json'));assert ex['returncode']!=0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    source=Path(m.__file__).read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fit');reads={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)};local={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Param))}|{a.arg for a in fn.args.args};missing=reads-local-set(m.__dict__)-set(dir(builtins));assert not missing,missing
    fake=dict(case='admission_only',epoch=1,train=.4,val=.4,bNLL=2.5,alphas=[.8,.8],seconds=1.);buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):print(m.json.dumps(fake),flush=True)
    assert json.loads(buffer.getvalue())==fake
    # Both actual routines still have all60 epochs and the same update graph.
    assert 'range(1,cfg[\'maximum_epochs\']+1)' in source and m.config()['maximum_epochs']==60
    m.claim(out.with_name(out.name+'.claim.json'),'independent_logging_repair_CPU');out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,old_CPU_proof_sha256=m.s.sha(m.HERE/'runs/smoke/caltech20_equal60_cpu_20261017/checks.json'),old_failure_sha256=failures,old_own_gate_sha256=oldgate,fit_global_names_resolved=True,actual_log_serialization_passed=True,maximum_epochs=60,new_model_objects=0,forward_calls=0,optimizer_steps=0,TEST_images_read=0,TEST_model_evaluations=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
