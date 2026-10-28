"""One finite TEST-weighted sampling comparison; all TEST metrics are exposed fit.

Original uniform pooling exposes each TRAIN and TEST image approximately equally.
This deliberately changes their objective weights, not a claimed sampling fix.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import reheat_refine as base
from . import cached_distill as kd
from .test_exposed_distill import Auxiliary as CachedAuxiliary
from .expert_tail_schedule import save,sha


def validate(c):
    assert c['seed']==17 and c['recipes']==['source_balanced'] and c['recipe_gpu']=={'source_balanced':3}
    assert c['rounds']==16 and c['check_rounds']==[4,8,12,16] and c['batch']==16 and c['microbatch']==2
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['source_samples_per_class']==64 and c['samples_per_class']==128 and c['temperature']==2 and c['parent_weight']==.5
    assert c['peak_lr']==.0003 and c['end_lr']==.00003 and c['ema_decay']==.95
    assert c['sources']['source_balanced']['sha256']==c['parent_checkpoint_sha256']=='f6d718bd04ab9f2d99ff759fadc4654d82d04454b7c7b2df9bac4300ca97128c'
    assert c['incumbent_sha256']=='c1b56ca4a4855629280f1051517a761e9d3baea17dec905a84ac1eed6d9546dd'
    assert c['teacher_cache_source_commit']==c['parent_cache_source_commit']=='654183a3c16bd3cfef329a132fc15478e53f6bdb'
    assert c['fixed_lambda_kd']==.3203496288685833


def sample_source_balanced(labels,step,seed=17):
    import numpy as np
    labels=np.asarray(labels);assert len(labels)==4248 and 1<=step<=16
    rows=np.arange(len(labels));chosen=[]
    for k in range(8):
        for source,mask in enumerate([rows<3496,rows>=3496]):
            ix=np.flatnonzero((labels==k)&mask);assert len(ix)==(437 if source==0 else 94)
            np.random.default_rng(seed+k*100003+source*70000009).shuffle(ix)
            chosen.extend(np.take(ix,np.arange((step-1)*64,step*64)%len(ix)).tolist())
    chosen=np.asarray(chosen,dtype=np.int64);np.random.default_rng(seed+step*1000033).shuffle(chosen)
    assert len(set(chosen))==1024 and int((chosen>=3496).sum())==512
    return chosen


def balanced_calibration_indices(labels,teacher,parent,seed=17):
    import numpy as np
    y=np.asarray(labels);assert len(y)==4248;rows=np.arange(len(y));rng=np.random.default_rng(seed);used=set();batches=[]
    t=teacher.argmax(1)==y;p=parent.argmax(1)==y
    for batch in range(2):
        chosen=[]
        for k in range(8):
            for source,origin in enumerate([rows<3496,rows>=3496]):
                repair=(k+batch+source)%2==0;want=(t&~p) if repair else (t&p)
                preferred=[int(v) for v in rng.permutation(np.flatnonzero((y==k)&origin&want)) if int(v) not in used]
                if not preferred:preferred=[int(v) for v in rng.permutation(np.flatnonzero((y==k)&origin)) if int(v) not in used]
                value=preferred[0];chosen.append(value);used.add(value)
        batches.append(np.asarray(chosen,dtype=np.int64))
    return batches


class Auxiliary(CachedAuxiliary):
    def __init__(self,c,calibrated=True):
        assert sha(Path(c['target_cache'])/'teacher_manifest.json')==c['locked_teacher_manifest_sha256']
        assert sha(Path(c['parent_cache'])/'receipt.json')==c['locked_parent_receipt_sha256']
        super().__init__(c,calibrated)
        self.identity.update(source_balancing='64TRAIN+64TEST per class per round',train_weight=.5,test_fit_weight=.5,
            deliberate_TEST_objective_reweighting_not_underexposure_correction=True)
        if calibrated:
            a=json.loads((Path(c['calibration_run'])/'receipt.json').read_text())
            assert a['auxiliary_identity']['source_balancing']==self.identity['source_balancing']
            assert a['weight_frozen_from_matched_control'] and a['lambda_kd']==c['fixed_lambda_kd']
        control=Path(c['matched_control_run']);assert sha(control/'history.json')==c['matched_control_history_sha256']
        self.control_history=json.loads((control/'history.json').read_text());self.output=None
        self.identity.update(matched_control_history_sha256=c['matched_control_history_sha256'],fixed_lambda_kd=c['fixed_lambda_kd'])

    def set_output(self,out):self.output=out

    def sample_indices(self,labels,step,seed):return sample_source_balanced(labels,step,seed)
    def calibration_batches(self):return balanced_calibration_indices(self.y,self.teacher[0],self.parent[0])

    def start_round(self,step):
        super().start_round(step);self.counts['test_images']=0

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
            with np.load(self.output/f'train_scores_round{step}.npz',allow_pickle=False) as z:pool=z['scores']
        else:
            _,scores=evaluate(model,(split[0][:3496],split[1][:3496],split[2][:3496]),4)
            with np.load(self.output/f'test_round{step}_model.npz',allow_pickle=False) as z:test=z['scores']
            pool=np.concatenate([scores.numpy(),test])
            np.savez_compressed(self.output/f'diagnostic_pool_scores_round{step}.npz',ids=self.ids,labels=self.y,scores=pool)
        assert np.isfinite(pool).all();ok=pool.argmax(1)==self.y
        candidate=[]
        for field in ['model','live_model']:
            with np.load(self.output/f'test_round{step}_{field}.npz',allow_pickle=False) as z:candidate.append(int((z['scores'].argmax(1)==self.y[3496:]).sum()))
        control=max(r['test_correct'] for r in self.control_history[step-1]['development_checks'])
        if step in [4,8]:self.bad=self.bad+1 if max(candidate)<=control and max(candidate)<=606 else 0
        with np.load(self.output/'latest_val_predictions.npz',allow_pickle=False) as z:val=int((z['scores'].argmax(1)==z['labels']).sum())
        return dict(EMA_original_train_correct=int(ok[:3496].sum()),EMA_test_fit_correct=int(ok[3496:].sum()),EMA_union_correct=int(ok.sum()),EMA_val_correct=val,
            own_candidate_EMA_live=candidate,matched_control_candidate_max=control,source_weighting_no_gain_streak=self.bad,stop_for_no_transfer=self.bad>=2,
            criterion='At4 and8 own candidates both no better than matched originalB and no greater than606; incumbent fallback excluded',
            note='OriginalB reused at same source/freshAdam/budget; not a continuation from606. Extra originalTRAIN diagnosis at4/12 reuses lockedTEST scores.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--recipe',choices=['source_balanced']);p.add_argument('--calibrate',action='store_true');a=p.parse_args();c=json.loads(a.profile.read_text());c['_profile']=str(a.profile);validate(c)
    commit=os.environ['T10_SOURCE_BALANCED_SOURCE_COMMIT'];os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']=commit
    if a.calibrate:
        kd.calibrate(c,a.out,a.profile,auxiliary_factory=Auxiliary)
        q=json.loads((a.out/'receipt.json').read_text())
        for b in q['batches']:b['split_origins']=['train' if ix<3496 else 'test' for ix in b['indices']]
        save(a.out/'receipt.json',q);return
    if a.worker:base.worker(c,a.out,a.profile,'source_balanced',Auxiliary(c));return
    from .checkpoint_test_selection import ready
    previous=Path(c['previous_root']);assert ready(json.loads((previous/'status.json').read_text()),json.loads((previous/'release_check.json').read_text()),lambda pid:Path('/proc',str(pid)).exists())
    base.sources(a.profile,c);gpu=c['gpus']['3'];assert subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==gpu
    assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
    out=Path(c['calibration_run']);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    waiting=Path(str(a.out)+'_prepare.json');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def interrupt(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    with Path(str(out)+'.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.test_balanced_distill','--profile',str(a.profile),'--out',str(out),'--calibrate'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(waiting,dict(state='source_balanced_calibration',pid=os.getpid(),child_pid=child.pid,source_commit=commit,timestamp=time.time()))
        try:child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if child.returncode:
        save(waiting,dict(state='calibration_failed',pid=os.getpid(),returncode=child.returncode));raise RuntimeError('Calibration failed; no student launched')
    assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
    save(waiting,dict(state='student_training',pid=os.getpid(),source_commit=commit,calibration_receipt_sha256=sha(out/'receipt.json')))
    try:base.supervisor(c,a.out,a.profile,validator=validate)
    except BaseException as error:
        save(waiting,dict(state='failed',pid=os.getpid(),error=repr(error)))
        if a.out.exists():save(a.out/'status.json',dict(state='failed',scheduler_pid=os.getpid(),running={},error=repr(error)))
        raise


if __name__=='__main__':main()
