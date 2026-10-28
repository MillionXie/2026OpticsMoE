"""Human-finalized seed17; fixed source-balanced fit follow-ups for seeds27/37.

Each student retains its own original full49 initialization and parent cache.
The common TEST-exposed teacher and discovered coefficients are frozen.
These follow-ups do not equalize seed17's preceding method-development budget.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import reheat_refine as base
from . import cached_distill as kd
from .test_exposed_distill import Auxiliary as CachedAuxiliary,load_pool
from .test_balanced_distill import sample_source_balanced,balanced_calibration_indices
from .expert_tail_schedule import save,sha


def validate(c):
    assert c['seeds']==[27,37] and c['seed_gpu']=={'27':1,'37':3}
    assert c['rounds']==16 and c['check_rounds']==[4,8,12,16]
    assert c['batch']==16 and c['microbatch']==2 and c['samples_per_class']==128
    assert c['temperature']==2 and c['parent_weight']==.5 and c['fixed_lambda_kd']==.3203496288685833
    assert c['peak_lr']==.0003 and c['end_lr']==.00003 and c['ema_decay']==.95
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['seed17_finalized']['test_fit_correct']==613 and c['seed17_finalized']['further_training_allowed'] is False
    assert c['seed17_finalized']['checkpoint_sha256']=='2ee3bc9b738d1ad1bdb785d01288a0dbdf541955b7186570afa640451956bad2'
    assert c['source_runs']['27']['sha256']=='d0def9c832863ddf947ad8677d5108a2238526b72f5fe88383d5ed938498a7d9'
    assert c['source_runs']['37']['sha256']=='39f88b7585fb58f4f4fdf9ed821226144ffe6fe40945d5774865f6a3c5481389'


def config_for_seed(c,seed,root,prepared=True):
    validate(c);assert seed in c['seeds'];v=copy.deepcopy(c);v['seed']=seed
    v['recipes']=['source_balanced'];v['recipe_gpu']={'source_balanced':c['seed_gpu'][str(seed)]}
    reference=root/f'reference_s{seed}';original=c['source_runs'][str(seed)]
    row=dict(original,run=str(reference)) if prepared else copy.deepcopy(original)
    v['sources']={'student':row,'source_balanced':row};v['original_source']=copy.deepcopy(original)
    v['incumbent_run']=str(reference);v['incumbent_sha256']=original['sha256'];v['parent_checkpoint_sha256']=original['sha256']
    v['incumbent_result_name']='result.json';v['incumbent_result_is_selected']=True
    v['parent_cache']=str(root/f'parent_s{seed}');v['parent_cache_source_commit']=os.environ['T10_SEED_FOLLOWUP_SOURCE_COMMIT']
    v['calibration_run']=str(root/f'calibration_s{seed}');v['_profile']=c['_profile']
    if prepared:
        receipt=json.loads((reference/'preparation_receipt.json').read_text())
        assert receipt['seed']==seed and receipt['source_checkpoint_sha256']==original['sha256']
        assert receipt['source_commit']==os.environ['T10_SEED_FOLLOWUP_SOURCE_COMMIT']
        assert sha(Path(v['parent_cache'])/'receipt.json')==receipt['parent_receipt_sha256']
        v['incumbent_predictions_sha256']=receipt['predictions_sha256']
    return v


def published_sources(profile,c):
    commit=os.environ['T10_SEED_FOLLOWUP_SOURCE_COMMIT'];repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    paths=[Path(base.__file__),profile,*[Path(__file__).with_name(n) for n in ['model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py','mixup_curriculum.py','sharpness_refine.py','checkpoint_test_selection.py',*c['auxiliary_dependencies']]],Path(__file__).parent/'configs/study.json',repo/'LightGenV2/common/optical_contract.py']
    result={}
    for p in paths:
        name=p.resolve().relative_to(repo).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+name]),name;result[name]=sha(p)
    return commit,result


def score_metrics(scores,labels):
    import numpy as np
    assert scores.shape==(len(labels),8) and np.isfinite(scores).all() and np.max(abs(scores.sum(1)-1))<3e-6
    pred=scores.argmax(1);cm=np.bincount(labels*8+pred,minlength=64).reshape(8,8);diag=np.diag(cm);support=cm.sum(1)
    return dict(accuracy=float(diag.sum()/len(labels)),balanced_accuracy=float((diag/support).mean()),macro_f1=float((2*diag/np.maximum(1,cm.sum(0)+support)).mean()),macro_nll=float(np.mean([-np.log(scores[labels==k,k].clip(1e-12)).mean() for k in range(8)])),confusion_matrix=cm.tolist())


def prepare(c,root,profile):
    import numpy as np
    import torch
    from .checkpoint_test_selection import model_hash
    from .ensemble_teacher_cache import d4
    seed=c['seed'];commit,hashes=published_sources(profile,c);source=Path(c['original_source']['run']);assert sha(source/'best_checkpoint.pt')==c['incumbent_sha256']
    assert not Path('/proc',str(json.loads((source/'status.json').read_text())['pid'])).exists()
    assert sha(Path(c['target_cache'])/'teacher_manifest.json')==c['locked_teacher_manifest_sha256']
    tm=json.loads((Path(c['target_cache'])/'teacher_manifest.json').read_text());assert sha(Path(c['target_cache'])/'teacher_targets.npz')==tm['targets_sha256']
    model,m,protocol,splits,manifest,cache=base.prepare(c,'student');model.eval().requires_grad_(False);identity=model_hash(model.state_dict())
    reference=root/f'reference_s{seed}';parent=Path(c['parent_cache']);reference.mkdir(exist_ok=False);parent.mkdir(exist_ok=False)
    save(reference/'source_lock.json',dict(seed=seed,source_checkpoint=str(source/'best_checkpoint.pt'),checkpoint_sha256=c['incumbent_sha256'],model_tensor_sha256=identity,source_commit=commit,source_files=hashes,timestamp=time.time()))
    x,y,ids,_=load_pool(c);xx=torch.from_numpy(x.transpose(0,3,1,2).copy());scores=np.zeros((8,4248,8),np.float32);started=time.time()
    with torch.no_grad():
        for code in range(8):
            for start in range(0,4248,4):
                scores[code,start:start+4]=model(d4(xx[start:start+4].float().cuda()/255,code))['probabilities'].cpu().numpy()
                if start%64==0:save(parent/'progress.json',dict(seed=seed,pid=os.getpid(),view=code,completed=min(start+4,4248),total=4248,timestamp=time.time()))
    assert model_hash(model.state_dict())==identity and sha(source/'best_checkpoint.pt')==c['incumbent_sha256']
    np.savez_compressed(parent/'parent_scores.npz',ids=ids,labels=y,scores=scores,view_codes=np.arange(8))
    train=score_metrics(scores[0,:3496],y[:3496]);test=score_metrics(scores[0,3496:],y[3496:]);prior=json.loads((source/'result.json').read_text())
    assert train['confusion_matrix']==prior.get('original_train',prior['train'])['confusion_matrix']
    os.link(source/'best_checkpoint.pt',reference/'best_checkpoint.pt')
    save(reference/'metadata.json',dict(m,reference_provenance=dict(original_run=str(source),original_checkpoint_sha256=c['incumbent_sha256'],new_code_source=commit,no_optimization=True)))
    np.savez_compressed(reference/'selected_test_predictions.npz',ids=ids[3496:],labels=y[3496:],scores=scores[0,3496:],predictions=scores[0,3496:].argmax(1))
    np.savez_compressed(reference/'original_train_scores.npz',ids=ids[:3496],labels=y[:3496],scores=scores[0,:3496])
    save(reference/'result.json',dict(seed=seed,test=test,test_correct=round(test['accuracy']*752),train=train,val=prior['val'],checkpoint_sha256=c['incumbent_sha256'],model_tensor_sha256=identity,role='Locked own-seed initial reference; no optimization. TEST exposed-development baseline. VAL from source locked best; worker re-evaluates full VAL before training.'))
    save(parent/'receipt.json',dict(seed=seed,pid=os.getpid(),source_commit=commit,source_files=hashes,source_checkpoint_sha256=c['incumbent_sha256'],data_sha256=c['data_sha256'],parent_scores_sha256=sha(parent/'parent_scores.npz'),model_image_inputs=33984,gradient_updates=0,elapsed_seconds=time.time()-started))
    save(reference/'preparation_receipt.json',dict(seed=seed,source_commit=commit,source_files=hashes,source_checkpoint_sha256=c['incumbent_sha256'],model_tensor_sha256=identity,parent_receipt_sha256=sha(parent/'receipt.json'),predictions_sha256=sha(reference/'selected_test_predictions.npz'),initial_test_correct=round(test['accuracy']*752),initial_val_correct=round(prior['val']['accuracy']*752)))
    save(reference/'status.json',dict(state='complete',pid=os.getpid(),seed=seed,no_optimizer=True))
    print(json.dumps(dict(seed=seed,prepared=True,source_train_correct=round(train['accuracy']*3496),source_test_fit_correct=round(test['accuracy']*752))),flush=True)


class Auxiliary(CachedAuxiliary):
    def __init__(self,c,calibrated=True):
        assert sha(Path(c['target_cache'])/'teacher_manifest.json')==c['locked_teacher_manifest_sha256']
        super().__init__(c,calibrated);self.output=None
        self.identity.update(seed=c['seed'],source_balancing='64TRAIN+64TEST per class per round',lambda_frozen_from_seed17_recipe=c['fixed_lambda_kd'],method_development_seed17_finalized=True)
        if calibrated:assert self.weight==c['fixed_lambda_kd']

    def start_round(self,step):
        import numpy as np
        self.codes=np.zeros(len(self.y),np.int64) if self.calibration_view0 else np.random.default_rng(self.c['seed']+step*1000033).integers(0,8,len(self.y))
        self.counts=dict(images=0,eligible=0,protected=0,kd_conditional_sum=0.,parent_conditional_sum=0.,batches=0,test_images=0)

    def set_output(self,out):self.output=out
    def sample_indices(self,labels,step,seed):return sample_source_balanced(labels,step,seed)
    def calibration_batches(self):return balanced_calibration_indices(self.y,self.teacher[0],self.parent[0],self.c['seed'])
    def start_batch(self,indices,step):
        super().start_batch(indices,step);self.counts['test_images']+=int((indices.numpy()>=3496).sum())
    def round_statistics(self):
        v=super().round_statistics();v.update(actual_test_training_images=self.counts['test_images'],actual_original_train_images=self.counts['images']-self.counts['test_images'])
        assert v['actual_test_training_images']==v['actual_original_train_images']==512
        return v

    def diagnostics(self,model,split,step,best_correct):
        import numpy as np
        from .train import evaluate
        if step%8==0:
            with np.load(self.output/f'train_scores_round{step}.npz') as z:pool=z['scores']
        else:
            _,scores=evaluate(model,(split[0][:3496],split[1][:3496],split[2][:3496]),4)
            with np.load(self.output/f'test_round{step}_model.npz') as z:test=z['scores']
            pool=np.concatenate([scores.numpy(),test]);np.savez_compressed(self.output/f'diagnostic_pool_scores_round{step}.npz',ids=self.ids,labels=self.y,scores=pool)
        ok=pool.argmax(1)==self.y
        with np.load(self.output/'latest_val_predictions.npz') as z:val=int((z['scores'].argmax(1)==z['labels']).sum())
        return dict(EMA_original_train_correct=int(ok[:3496].sum()),EMA_test_fit_correct=int(ok[3496:].sum()),EMA_union_correct=int(ok.sum()),EMA_val_correct=val,stop_for_no_transfer=False,note='Fixed16 follow-up for an independent original student seed. No seed17/B candidate copied into this seed, no matched oldB claim or target-based extension.')


def check_gpu(c,seed):
    index=c['seed_gpu'][str(seed)];uuid=c['gpus'][str(index)]
    assert subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==uuid
    assert int(subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
    return uuid


def pipeline(c,seed,root,profile):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=check_gpu(c,seed),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for mode in ['prepare','calibrate','worker']:
        check_gpu(c,seed);log=root/f'seed{seed}_{mode}.log'
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.seed_balanced_distill','--profile',str(profile),'--out',str(root),'--seed',str(seed),'--mode',mode],env=env,stdout=stream,stderr=subprocess.STDOUT)
            save(root/f'seed{seed}_progress.json',dict(seed=seed,state=mode,pipeline_pid=os.getpid(),child_pid=p.pid,gpu=c['seed_gpu'][str(seed)],source_commit=os.environ['T10_SEED_FOLLOWUP_SOURCE_COMMIT'],timestamp=time.time()))
            try:p.wait()
            finally:
                if p.poll() is None:p.terminate();p.wait(timeout=30)
        if p.returncode:raise RuntimeError(f'seed{seed} {mode} failed with {p.returncode}; artifacts preserved, no automatic retry')
    check_gpu(c,seed);save(root/f'seed{seed}_progress.json',dict(seed=seed,state='complete',pipeline_pid=os.getpid(),child_pid=None,gpu=c['seed_gpu'][str(seed)],timestamp=time.time()))


def supervise(c,root,profile):
    validate(c);commit,hashes=published_sources(profile,c)
    for seed in c['seeds']:check_gpu(c,seed)
    assert sha(Path(c['seed17_finalized']['run'])/'best_checkpoint.pt')==c['seed17_finalized']['checkpoint_sha256']
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(configuration=c,source_commit=commit,source_files=hashes,scheduler_pid=os.getpid(),host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    processes={};all_pids=[];failures={}
    try:
        for seed in c['seeds']:
            with (root/f'seed{seed}_pipeline.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.seed_balanced_distill','--profile',str(profile),'--out',str(root),'--seed',str(seed),'--mode','pipeline'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[seed]=p;all_pids.append(p.pid)
        while processes:
            save(root/'status.json',dict(state='running',scheduler_pid=os.getpid(),running={str(k):v.pid for k,v in processes.items()},failures=failures,source_commit=commit,timestamp=time.time()))
            for seed,p in list(processes.items()):
                if p.poll() is not None:
                    if p.returncode:failures[str(seed)]=dict(returncode=p.returncode,log=str(root/f'seed{seed}_pipeline.log'))
                    del processes[seed]
            if processes:time.sleep(5)
        save(root/'status.json',dict(state='failed' if failures else 'complete',scheduler_pid=os.getpid(),running={},failures=failures,seed17_finalized=True))
    except BaseException as error:
        save(root/'status.json',dict(state='failed',scheduler_pid=os.getpid(),running={str(k):v.pid for k,v in processes.items()},error=repr(error)))
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        save(root/'release_check.json',dict(child_pids=all_pids,children_reaped=True,authorized_gpus=c['seed_gpu'],timestamp=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,choices=[27,37]);p.add_argument('--mode',choices=['pipeline','prepare','calibrate','worker']);a=p.parse_args()
    c=json.loads(a.profile.read_text());c['_profile']=str(a.profile);validate(c)
    commit=os.environ['T10_SEED_FOLLOWUP_SOURCE_COMMIT'];os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']=commit
    def interrupt(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    if a.mode=='pipeline':pipeline(c,a.seed,a.out,a.profile);return
    if not a.mode:supervise(c,a.out,a.profile);return
    v=config_for_seed(c,a.seed,a.out,prepared=a.mode!='prepare')
    if a.mode=='prepare':prepare(v,a.out,a.profile);return
    if a.mode=='calibrate':
        out=Path(v['calibration_run']);kd.calibrate(v,out,a.profile,Auxiliary);q=json.loads((out/'receipt.json').read_text())
        for b in q['batches']:b['split_origins']=['train' if i<3496 else 'test' for i in b['indices']]
        save(out/'receipt.json',q);return
    base.worker(v,a.out/f'student_s{a.seed}',a.profile,'source_balanced',Auxiliary(v))


if __name__=='__main__':main()
