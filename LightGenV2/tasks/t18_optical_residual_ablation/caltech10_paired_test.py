"""One human-authorized test of both frozen audited Caltech10 models."""
import argparse,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech10_gap as parent
import caltech10_experiment_resources as resources
from audit_resisc10 import predictions
s=parent.s;HERE=parent.HERE;ROOT=parent.ROOT
PROFILE=HERE/'caltech10_paired_test_profile.json'
FILES=('caltech10_paired_test.py','caltech10_paired_test_profile.json','CALTECH10_PAIRED_TEST_PROTOCOL_20261015.md','caltech10_experiment_resources.py')
CPU=HERE/'runs/smoke/caltech10_paired_test_refusal_20261015'
def sources():
    v=parent.sources()
    for n in FILES:v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return v
def barrier(base):
    p=s.read(PROFILE);assert base==HERE/'runs/simulation'/p['parent_run']
    meta=s.read(base/'metadata.json');state=s.read(base/'status.json');done=s.read(base/'audit_completed.json');campaign=s.read(base.with_name(base.name+'.campaign_claim.json'));q=s.read(base/'queue_exit.json')
    assert meta['git_commit']==p['parent_commit'] and meta['sources']==parent.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert state['state']=='caltech10_gap_complete_science_passed' and not state['failures'] and state['active']==state['pending']==0
    assert q['returncode']==0 and q['actual_exit'] and parent.base.exited(q['pid']) and parent.base.exited(campaign['pid']) and parent.base.exited(meta['pid'])
    assert done['passed'] and done['actual_audit_returncode']==0 and done['audit_sha256']==s.sha(base/'cpu_report/audit.json')==p['parent_audit_sha256']
    assert all(j['exit_code']==0 and parent.base.exited(j['pid']) for j in state['completed'])
    fits={}
    for key,rho in parent.CASES.items():
        arm=base/key;f=s.read(arm/'fit_result.json');h=s.read(arm/'history.json');assert f['epochs_completed']==150 and f['rho']==rho
        assert f['checkpoint_sha256']==s.sha(arm/'best_checkpoint.pt')==p['selected_checkpoint_sha256'][key] and f['selected_epoch']==min(h,key=lambda z:z['val']['balanced_nll'])['epoch']==149
        assert not any((arm/n).exists() for n in ('failure.json','cpu_report_failure.json'));fits[key]=f
    return fits
def refusal(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==CPU and not a.out.exists();barrier(a.base);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    try:barrier(a.base.with_name('wrong_run'))
    except AssertionError:pass
    else:raise AssertionError('Wrong run admitted')
    assert not (a.base/s.read(PROFILE)['output']).exists();s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,git_commit=commit,automatic_retry=False),exclusive=True);a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,wrong_run_rejected=True,models_constructed=0,forward_calls=0,test_arrays_read=0,test_evaluations=0),exclusive=True)
def evaluate(a):
    f=barrier(a.base)[a.case];assert a.gpu in resources.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    claim=s.read(a.out/'claim.json');assert claim['sources']==sources() and claim['checkpoint_sha256']==f['checkpoint_sha256'];s.engine.verify_sources(claim['sources'],claim['git_commit'])
    assert not any((a.out/n).exists() for n in ('test_started.json','result.json','test_predictions.csv'));resources.slots();s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    s.save(a.out/'test_started.json',dict(pid=os.getpid(),checkpoint_sha256=f['checkpoint_sha256'],started_unix=time.time(),maximum_evaluations=1,automatic_retry=False),exclusive=True)
    m=parent.manifest(Path(f['config']['data']))
    with np.load(f['config']['data'],allow_pickle=False) as z:x,y,ids=[z['test_'+n].copy() for n in ('images','labels','ids')]
    keep=np.isin(y,parent.INDICES);x=x[keep];y=parent.subset_labels(y);ids=ids[keep]
    assert x.shape==(1833,150,150,3) and ids.tolist()==m['split_ids']['test'] and np.bincount(y,minlength=10).tolist()==m['supports']['test']
    parent.bind();s.engine.seed(17);model=parent.SubsetMoE(4,f['rho']).cuda();ck=torch.load(a.base/a.case/'best_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==f['selected_epoch'];model.load_state_dict(ck['model'],strict=True)
    metric,rows=parent.evaluate(model,(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids),8)
    assert s.sha(a.base/a.case/'best_checkpoint.pt')==claim['checkpoint_sha256'];s.engine.csvwrite(a.out/'test_predictions.csv',rows)
    s.save(a.out/'result.json',dict(test=metric,test_evaluations=1,test_used_for_selection=False,selected_epoch=f['selected_epoch'],checkpoint_sha256=f['checkpoint_sha256'],sources=claim['sources'],git_commit=claim['git_commit'],classes=m['classes'],rho=f['rho'],electronic_classification_branch=False,model_forward_batches=math.ceil(1833/8),gpu_policy=policy,scope='Frozen validation-informed ten-class exploratory simulation test; prior Caltech test observed'),exclusive=True)
def saved_audit(base,out,key):
    f=barrier(base)[key];v=s.read(out/'result.json');claim=s.read(out/'claim.json');started=s.read(out/'test_started.json');exit_rec=s.read(out/'producer_exit.json')
    assert exit_rec['returncode']==0 and exit_rec['actual_exit'] and parent.base.exited(started['pid']) and exit_rec['pid']==started['pid']
    assert v['test_evaluations']==1 and not v['test_used_for_selection'] and v['checkpoint_sha256']==claim['checkpoint_sha256']==f['checkpoint_sha256'];assert v['sources']==claim['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
    m=parent.manifest(Path(f['config']['data']))
    with np.load(f['config']['data'],allow_pickle=False) as z:y=parent.subset_labels(z['test_labels'].copy())
    predictions(out/'test_predictions.csv',v['test'],m['split_ids']['test'],y)
    s.save(out/'saved_test_audit.json',dict(passed=True,result_sha256=s.sha(out/'result.json'),checkpoint_sha256=f['checkpoint_sha256'],independent_ID_label_argmax_CM_support_macro_NLL=True,model_forward_calls=0,test_image_arrays_read=0,test_label_arrays_read=1,test_evaluations_total=1),exclusive=True)
def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==a.base/s.read(PROFILE)['output'] and not a.out.exists()
    fits=barrier(a.base);proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();s.engine.verify_sources(proof['sources'],proof['git_commit']);resources.slots()
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'claim.json',dict(pid=os.getpid(),sources=src,git_commit=commit,checkpoints={k:f['checkpoint_sha256'] for k,f in fits.items()},classes=next(iter(fits.values()))['config']['classes'],selection='firstminimum val EMA balancedNLL frozen before either test',maximum_tests_per_model=1,automatic_retry=False),exclusive=True)
    jobs=[]
    for key,index in zip(parent.CASES,(2,4)):
        gpu=resources.GPU[index];live=resources.slots();assert live['total']<4 and live['counts'][gpu]==0;s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
        out=a.out/key;out.mkdir(exist_ok=False);s.save(out/'claim.json',dict(sources=src,git_commit=commit,checkpoint_sha256=fits[key]['checkpoint_sha256'],maximum_tests=1,automatic_retry=False),exclusive=True);log=(out/'test.log').open('x')
        child=subprocess.Popen([sys.executable,str(HERE/'caltech10_paired_test.py'),'--phase','evaluate','--base',str(a.base),'--out',str(out),'--case',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append((key,out,child,log))
    for key,out,child,log in jobs:
        rc=child.wait();log.close();s.save(out/'producer_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=parent.base.exited(child.pid)),exclusive=True);assert rc==0 and parent.base.exited(child.pid),'Test failed: preserve no retry';saved_audit(a.base,out,key)
    results={k:s.read(out/'result.json') for k,out,_,_ in jobs};vals=list(results.values());gap=vals[0]['test']['accuracy']-vals[1]['test']['accuracy'];macro=vals[0]['test']['balanced_accuracy']-vals[1]['test']['balanced_accuracy']
    s.save(a.out/'paired_result.json',dict(results=results,residual_minus_control_pp=100*gap,macro_difference_pp=100*macro,target_gap10pp_achieved=gap>=.1,test_used_for_selection=False,test_evaluations_per_model=1),exclusive=True)
    s.save(a.out/'completed.json',dict(passed=True,paired_result_sha256=s.sha(a.out/'paired_result.json'),saved_audit_sha256={k:s.sha(out/'saved_test_audit.json') for k,out,_,_ in jobs},test_evaluations_per_model=1),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['refusal','queue','evaluate'],required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists() and not (a.out/'failure.json').exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
