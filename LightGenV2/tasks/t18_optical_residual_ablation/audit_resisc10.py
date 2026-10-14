"""Independent10-label saved science, no forward or test arrays."""
import argparse,copy,csv,inspect,math,os,textwrap
from pathlib import Path
import numpy as np
import resisc10 as study
import audit_resisc20 as sealed
from audit_caltech_repair234 import exited
from audit_resisc20_input_pair import scheduler_budget_checks
s=study.s
def predictions(path,metric,ids,labels):
    rows=list(csv.DictReader(path.open()));assert [r['sample_id'] for r in rows]==list(ids)
    y=np.asarray([int(r['label_true']) for r in rows]);assert np.array_equal(y,labels)
    p=np.asarray([[float(r[f'score{k}']) for k in range(10)] for r in rows]);assert np.array_equal(p.argmax(1),[int(r['label_pred']) for r in rows])
    pred=p.argmax(1);support=np.bincount(y,minlength=10);assert p.shape==(len(y),10) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,pred),1);recall=cm.diagonal()/support;nll=-np.log(p[np.arange(len(y)),y].clip(1e-12))
    calculated=dict(n=len(y),accuracy=float((pred==y).mean()),balanced_accuracy=float(recall.mean()),recall=recall.tolist(),support=support.tolist(),confusion_matrix=cm.tolist(),nll=float(nll.mean()),balanced_nll=float(np.mean([nll[y==k].mean() for k in range(10)])))
    for k,v in calculated.items():
        if isinstance(v,float):assert math.isclose(v,metric[k],rel_tol=1e-9,abs_tol=1e-9)
        else:assert v==metric[k]
def verify(base,key,data,m):
    source=textwrap.dedent(inspect.getsource(sealed.verify))
    changes={"<=30":"<=cfg['maximum_epochs']","s.r.CorrectedMoE(4,f['rho'])":"study.SubsetMoE(4,f['rho'])", "labels=z[role+'_labels'].copy()":"labels=study.subset_labels(z[role+'_labels'].copy())",'s.eligibility(f,hist)':'study.eligibility(f,hist)'}
    for old,new in changes.items():assert source.count(old)==1;source=source.replace(old,new)
    ns=dict(sealed.__dict__);ns.update(study=study,predictions=predictions);exec(compile(source,__file__+':sealed10savedverify','exec'),ns)
    f=ns['verify'](base,key,data,m);expected=study.config();expected.update(depth=4,residual_rho=f['rho'],classes=m['classes'],contract=study.contract(4,f['rho']));assert f['config']==expected
    cfg=f['config'];assert cfg['class_count']==10 and cfg['original_indices']==list(study.INDICES) and cfg['subset_manifest_sha256']==s.sha(study.private_manifest())
    h=s.read(base/key/'history.json');last=__import__('torch').load(base/key/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    budget=scheduler_budget_checks(cfg,h,last,f['stop_reason']);return f,budget
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and a.base.name==study.RUN
    study.bind();m=study.manifest(a.data);meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json')
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==2
    s.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['sources']==study.sources();fits={};budgets={}
    for j in state['completed']:
        assert exited(j['pid']) and j['exit_code']==0 and j['gpu_uuid'] in study.GPU.values()
        f,b=verify(a.base,j['case'],a.data,m);assert f['rho']==study.CASES[j['case']] and f['sources']==meta['sources'];fits[j['case']]=f;budgets[j['case']]=b
    assert set(fits)==set(study.CASES) and len({f['initial_parameter_sha256'] for f in fits.values()})==1
    cfgs=[]
    for f in fits.values():
        v=copy.deepcopy(f['config']);v.pop('residual_rho');v.pop('contract');cfgs.append(v)
    assert cfgs[0]==cfgs[1];common=min(f['epochs_completed'] for f in fits.values());assert len({tuple(f['orders'][:common]) for f in fits.values()})==len({tuple(f['transforms'][:common]) for f in fits.values()})==1
    gates={k:s.read(a.base/k/'validation_gate.json') for k in fits};a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'audit.json',dict(passed=True,fits=fits,gates=gates,budget_checks=budgets,all_pair_eligible=all(v['passed'] for v in gates.values()),class_count=10,original20_target_achieved=False,train_samples=5600,val_samples=700,optimizer_steps_per_epoch=350,tail_batch=16,common_pair_epochs=common,training_source_commit=meta['git_commit'],sources=meta['sources'],audit_source_sha256=s.sha(__file__),cpu_models_constructed_for_shape_seed=2,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,formal_authorized_by_this_receipt=False),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out','data'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        mark=a.base/'cpu_report_failure.json'
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
