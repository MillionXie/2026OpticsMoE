"""Input-case-aware saved CPU review of all three fresh controls; no forward/test."""
import argparse,inspect,os,textwrap
from pathlib import Path
import resisc20 as s
import resisc20_input_study as study
import audit_resisc20 as old
from audit_caltech_repair234 import exited

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and not (a.base/'cpu_report_failure.json').exists()
    study.prerequisites();m=s.manifest(a.data);meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json')
    assert meta['sources']==study.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==3
    source=textwrap.dedent(inspect.getsource(old.verify));assert source.count("f['epochs_completed']<=30")==1
    ns=dict(old.__dict__);exec(compile(source.replace("f['epochs_completed']<=30","f['epochs_completed']<=cfg['maximum_epochs']"),str(__file__)+':independent_saved_verify','exec'),ns)
    fits={}
    for j in state['completed']:
        assert exited(j['pid']) and j['exit_code']==0 and j['case'] in study.CASES
        f=ns['verify'](a.base,j['case'],a.data,m);tile,rho=study.CASES[j['case']]
        expected=study.config(j['case']);expected.update(depth=4,residual_rho=rho,classes=m['classes'],contract=study.contract(4,rho,tile))
        assert f['config']==expected
        assert f['sources']==meta['sources'] and f['config']['study_case']==j['case'] and f['config']['input_tile']==tile and f['rho']==rho
        assert f['config']['contract']==study.contract(4,rho,tile) and f['config']['maximum_epochs']==120 and f['config']['maximum_seconds']==7200 and f['config']['microbatch']==8
        fits[j['case']]=f
    assert set(fits)==set(study.CASES) and len({f['initial_parameter_sha256'] for f in fits.values()})==1
    normalized=[]
    for f in fits.values():
        cfg=dict(f['config'])
        for k in ('study_case','input_tile','residual_rho','contract'):cfg.pop(k)
        normalized.append(cfg)
    assert all(v==normalized[0] for v in normalized)
    common=min(f['epochs_completed'] for f in fits.values())
    assert len({tuple(f['orders'][:common]) for f in fits.values()})==len({tuple(f['transforms'][:common]) for f in fits.values()})==1
    gates={k:s.read(a.base/k/'validation_gate.json') for k in fits}
    eligible=[k for k in ('L4_tile50_rho03','L4_tile100_rho03') if gates[k]['passed']]
    chosen=min(eligible,key=lambda k:fits[k]['metrics']['val']['balanced_nll']) if eligible else None
    a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'audit.json',dict(passed=True,fits=fits,gates=gates,eligible_residual_cases=eligible,chosen_residual_case=chosen,
        input100_pair_eligible=all(gates[k]['passed'] for k in ('L4_tile100_rho03','L4_tile100_rho0')),
        sources=meta['sources'],training_source_commit=meta['git_commit'],shared_prefix_epochs=common,
        train_samples=11198,optimizer_steps_per_epoch=700,tail_batch=14,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,
        scope='Fresh controlled input50vs100 at rho.3 and input100 rho pair; trained native and independent2/3 still required, no automatic formal admission'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','out','data'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        marker=a.base/'cpu_report_failure.json'
        if not marker.exists():s.save(marker,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
