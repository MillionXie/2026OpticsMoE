"""Independent10-class saved-result guard and immutable renderer adaptation."""
import argparse,inspect,json,os,textwrap
from pathlib import Path
import numpy as np
import resisc10_optimize as study
import audit_resisc10_optimize as audit
import plot_saved_validation_studies as render
from audit_caltech_repair234 import exited
s=study.s;KIND='resisc10_optimize'
RENDER_SHA='bbce2491d452389667339c4fef1ad53c8bc07b941a7dce7672845281c9489ebb'
SPEC=dict(run=study.RUN,commit=None,source_count=165,keys=tuple(study.CASES),labels=('10 classes rho0.3','10 classes rho0'),parameters=(908552,908552),train_n=5600,val_n=700,budget=120,title='RESISC validation-selected top10: pure optical MixUp .2 full-size L4 rho pair, changed classification task')
def guarded(kind,base):
    assert kind==KIND and base.name==study.RUN
    proof=s.read(base/'cpu_report/audit.json');dispatch=s.read(base/'saved_cpu_audit_dispatch_completed_20261015.json');claim=s.read(base/'saved_cpu_audit_dispatch_20261015.json')
    assert proof['passed'] and dispatch['passed'] and dispatch['audit_sha256']==s.sha(base/'cpu_report/audit.json')
    assert proof['class_count']==10 and not proof['original20_target_achieved']
    assert proof['audit_source_sha256']==dispatch['audit_source_sha256']==claim['audit_source_sha256']==s.sha(audit.__file__)
    assert proof['model_forward_calls']==proof['test_arrays_read']==proof['test_evaluations']==0 and not (base/'cpu_report_failure.json').exists()
    meta=s.read(base/'metadata.json');state=s.read(base/'status.json');cur=s.read(base/'current_processes.json')
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and not cur['active'] and not cur['pending'] and len(state['completed'])==2
    assert {j['case'] for j in state['completed']}==set(study.CASES) and all(j['exit_code']==0 and exited(j['pid']) and j['gpu_uuid'] in study.GPU.values() for j in state['completed'])
    assert meta['sources']==proof['sources']==study.sources() and len(meta['sources'])==165
    spec=dict(SPEC,commit=meta['git_commit'])
    assert meta['git_commit']==proof['training_source_commit']==dispatch['source_commit']==claim['source_commit']==spec['commit']
    s.engine.verify_sources(meta['sources'],meta['git_commit']);cfg=study.config();m=study.manifest(Path(cfg['data']));assert len(m['classes'])==10
    data={}
    for key,rho in study.CASES.items():
        f=s.read(base/key/'fit_result.json');hist=s.read(base/key/'history.json');expected=dict(cfg,depth=4,residual_rho=rho,classes=m['classes'],contract=study.contract(4,rho))
        assert f==proof['fits'][key] and f['config']==expected and f['sources']==meta['sources'] and f['rho']==rho
        assert f['epochs_completed']==len(hist)<=120 and f['metrics']['train']['n']==5600 and f['metrics']['val']['n']==700
        assert f['test_evaluations']==0 and not f['test_used_for_selection'] and not (base/key/'failure.json').exists()
        assert s.sha(base/key/'history.json')==f['history_sha256'] and s.sha(base/key/'best_checkpoint.pt')==f['checkpoint_sha256']
        assert min(hist,key=lambda h:h['val']['balanced_nll'])['epoch']==f['selected_epoch']
        assert proof['gates'][key]==s.read(base/key/'validation_gate.json') and proof['budget_checks'][key]['passed']
        for role in ('train','val'):
            metric=f['metrics'][role];cm=np.asarray(metric['confusion_matrix']);support=np.asarray(m['supports'][role])
            assert cm.shape==(10,10) and np.array_equal(cm.sum(1),support) and metric['support']==support.tolist()
            assert len(metric['recall'])==len(metric['routing']['per_class_weight_mean'])==10
        data[key]=(f,hist)
    common=min(f['epochs_completed'] for f,_ in data.values());assert common==proof['common_pair_epochs'] and proof['all_pair_eligible']==all(proof['gates'][k]['passed'] for k in study.CASES)
    assert len({f['initial_parameter_sha256'] for f,_ in data.values()})==len({tuple(f['orders'][:common]) for f,_ in data.values()})==len({tuple(f['transforms'][:common]) for f,_ in data.values()})==1
    return spec,proof,data
def renderer_entry():
    assert s.sha(render.__file__)==RENDER_SHA
    source=textwrap.dedent(inspect.getsource(render.plot));assert source.count('assert len(classes)==20')==1 and source.count('range(20)')==5
    source=source.replace('assert len(classes)==20','assert len(classes)==10').replace('range(20)','range(10)')
    ns=dict(render.__dict__);ns.update(guarded=guarded,__file__=__file__);exec(compile(source,__file__+':sealed10savedrender','exec'),ns)
    return ns['plot']
def plot(base,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and out==base/'cpu_report/figures' and not out.exists()
    guarded(KIND,base);receipt=renderer_entry()(KIND,base,out)
    receipt.update(class_count=10,original20_target_achieved=False,generic_renderer_sha256=RENDER_SHA,renderer_adaptation='One class-count assertion and five fixed class-axis/row loop bounds20->10; all metric rendering unchanged.',scope='Genuine audited saved10-class EMA only; classification task changed. Single seed and measured actual epochs/budgets; no original20 success or test evidence.')
    assert receipt['pdf_count']==4 and len(list(out.glob('*.pdf')))==4
    (out/'artifact_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists(),'Preserve existing/failed figures'
    try:print(json.dumps(plot(a.base,a.out)))
    except BaseException as e:
        if a.out.exists():
            with (a.out/'failure.json').open('x',encoding='utf-8') as f:json.dump(dict(error=repr(e),automatic_retry=False),f)
        raise
