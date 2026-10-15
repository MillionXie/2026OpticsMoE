"""One frozen residual test after both new stages and independent saved audits."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import resisc10_continue100 as continuation
from audit_caltech_repair234 import exited
from audit_resisc10 import predictions
s=continuation.s;base=continuation.base;HERE=base.HERE;ROOT=base.ROOT
FILES=('resisc10_final_test.py','RESISC10_FINAL_TEST_PROTOCOL_20261015.md')
def sources():
    v=continuation.sources()
    for name in FILES:v[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    assert len(v)==171;return v
def barrier(arm):
    p=continuation.profile();root=HERE/'runs/simulation'/p['run'];assert arm==root/'regularize20'
    meta=s.read(root/'metadata.json');status=s.read(root/'status.json');assert exited(meta['pid']) and status['state']=='complete'
    assert meta['sources']==continuation.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
    for stage in ('complete100','regularize20'):
        b=root/stage;done=s.read(b/'audit_completed.json');proof=s.read(b/'cpu_report/audit.json');exit_rec=s.read(b/'producer_exit.json')
        assert done['passed'] and proof['passed'] and done['audit_sha256']==s.sha(b/'cpu_report/audit.json') and done['actual_audit_returncode']==0
        assert exit_rec['actual_exit'] and exit_rec['returncode']==0 and exited(exit_rec['pid']) and not (b/'failure.json').exists() and not (b/'cpu_report_failure.json').exists()
    f=s.read(arm/'fit_result.json');assert f['rho']==.3 and f['epochs_completed']==120 and f['test_evaluations']==0
    assert f['checkpoint_sha256']==s.sha(arm/'best_checkpoint.pt') and f['selected_epoch']==min(s.read(arm/'history.json'),key=lambda h:h['val']['balanced_nll'])['epoch']
    return f
def evaluate(a):
    assert a.gpu==base.GPU[2] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    f=barrier(a.base);claim=s.read(a.out/'claim.json');assert claim['checkpoint_sha256']==f['checkpoint_sha256'] and claim['sources']==sources()
    s.engine.verify_sources(claim['sources'],claim['git_commit']);assert not any((a.out/name).exists() for name in ('test_started.json','result.json','test_predictions.csv'))
    base.slots();s.gpu_budget.ALLOWED.update(base.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    s.save(a.out/'test_started.json',dict(pid=os.getpid(),checkpoint_sha256=f['checkpoint_sha256'],started_unix=time.time(),test_evaluations_authorized=1,automatic_retry=False),exclusive=True)
    m=base.manifest(Path(f['config']['data']));assert m['classes']==f['config']['classes']
    with np.load(f['config']['data'],allow_pickle=False) as z:x,y,ids=[z['test_'+k].copy() for k in ('images','labels','ids')]
    keep=np.isin(y,base.INDICES);x=x[keep];y=base.subset_labels(y);ids=ids[keep]
    assert x.dtype==np.uint8 and x.shape==(700,150,150,3) and ids.tolist()==m['split_ids']['test'] and np.bincount(y,minlength=10).tolist()==[70]*10
    data=(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids)
    base.bind();s.engine.seed(17);model=base.SubsetMoE(4,.3).cuda();ck=torch.load(a.base/'best_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==f['selected_epoch'];model.load_state_dict(ck['model'],strict=True)
    metric,rows=base.evaluate(model,data,8);assert s.sha(a.base/'best_checkpoint.pt')==claim['checkpoint_sha256']
    s.engine.csvwrite(a.out/'test_predictions.csv',rows)
    s.save(a.out/'result.json',dict(test=metric,test_evaluations=1,test_used_for_selection=False,test_image_arrays_read=1,selected_epoch=f['selected_epoch'],checkpoint_sha256=f['checkpoint_sha256'],parent_fit_sha256=s.sha(a.base/'fit_result.json'),parent_audit_sha256=s.sha(a.base/'cpu_report/audit.json'),sources=claim['sources'],git_commit=claim['git_commit'],classes=m['classes'],class_count=10,optical_rho=.3,electronic_classification_branch=False,gpu_policy=policy,model_forward_batches=88,target80_achieved=metric['accuracy']>=.8,scope='Single frozen test on validation-informed top10 task, not original20 acceptance'),exclusive=True)
def saved_audit(a):
    f=barrier(a.base);v=s.read(a.out/'result.json');claim=s.read(a.out/'claim.json');started=s.read(a.out/'test_started.json');exit_rec=s.read(a.out/'test_producer_exit.json')
    assert exit_rec['returncode']==0 and exit_rec['actual_exit'] and exited(started['pid']) and exit_rec['pid']==started['pid']
    assert v['test_evaluations']==1 and not v['test_used_for_selection'] and v['checkpoint_sha256']==claim['checkpoint_sha256']==f['checkpoint_sha256']
    assert v['sources']==claim['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
    m=base.manifest(Path(f['config']['data']))
    with np.load(f['config']['data'],allow_pickle=False) as z:y=base.subset_labels(z['test_labels'].copy())
    predictions(a.out/'test_predictions.csv',v['test'],m['split_ids']['test'],y)
    s.save(a.out/'saved_test_audit.json',dict(passed=True,result_sha256=s.sha(a.out/'result.json'),checkpoint_sha256=f['checkpoint_sha256'],independent_ID_label_argmax_CM_support_NLL_verified=True,model_forward_calls=0,test_image_arrays_read=0,test_label_arrays_read=1,test_evaluations_total=1),exclusive=True)
def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==a.base.parent/'final_once_test' and not a.out.exists()
    f=barrier(a.base);base.slots();s.gpu_budget.ALLOWED.update(base.GPU.values());s.gpu_budget.preflight(base.GPU[2],shared=True,memory_gib=2)
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'claim.json',dict(pid=os.getpid(),checkpoint_sha256=f['checkpoint_sha256'],parent_audit_sha256=s.sha(a.base/'cpu_report/audit.json'),sources=src,git_commit=commit,selection='first minimum validation EMA balancedNLL; frozen before test',authorization='Human20261015 explicitly identifies test accuracy as ultimate target',maximum_test_evaluations=1,automatic_retry=False),exclusive=True)
    with (a.out/'test.log').open('x') as log:
        cmd=[sys.executable,str(HERE/'resisc10_final_test.py'),'--phase','evaluate','--base',str(a.base),'--out',str(a.out),'--gpu',base.GPU[2]]
        child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU[2]),stdout=log,stderr=subprocess.STDOUT);rc=child.wait()
    s.save(a.out/'test_producer_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=exited(child.pid)),exclusive=True);assert rc==0 and exited(child.pid),'Test failed: preserve claim, no retry'
    saved_audit(a);s.save(a.out/'completed.json',dict(passed=True,result_sha256=s.sha(a.out/'result.json'),saved_audit_sha256=s.sha(a.out/'saved_test_audit.json'),test_evaluations=1),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['queue','evaluate'],required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists() and not (a.out/'failure.json').exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
