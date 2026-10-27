"""Finite TRAIN teacher-complement supervision with full parent-correct retention."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import cached_distill as kd
from . import reheat_refine as base
from .expert_tail_schedule import save,sha


def validate(c):
    assert c['seed']==17 and c['rounds']==16 and c['recipes']==['student']
    assert c['check_rounds']==[4,8,12,16] and c['batch']==16 and c['microbatch']==2 and c['extra_batch']==4
    assert c['samples_per_class']==128 and c['schedule_type']=='distill' and c['peak_lr']==.0003 and c['end_lr']==.00003
    assert c['ema_decay']==.95 and c['target_correct']==616 and c['parent_weight']==.5 and c['temperature']==1
    assert c['sources']['student']['sha256']==c['incumbent_sha256']=='75e51f4e117f451ae7eb7947f0d001af0c07155450280fea3b19d9e9970d55a7'
    assert c['recipe_gpu']=={'student':3} and c['minimum_weighted_gradient_ratio']==.05


def complement_masks(teacher,parent,labels):
    return (teacher.argmax(-1)==labels)&(parent.argmax(-1)!=labels),parent.argmax(-1)==labels


def pick_extra(pool,offset,excluded):
    """Four same-view complement rows; IDs distinct and disjoint from main batch."""
    selected=[];used=set(int(v) for v in excluded)
    for j in range(len(pool)):
        code,index=pool[(offset+j)%len(pool)]
        if int(index) in used:continue
        selected.append((int(code),int(index)));used.add(int(index))
        if len(selected)==4:return selected
    raise ValueError('Insufficient distinct TRAIN complement IDs')


class Auxiliary(kd.Auxiliary):
    def __init__(self,c,calibrated=True):
        import numpy as np
        self.c=c;self.root=Path(c['target_cache']);manifest=json.loads((self.root/'cache_manifest.json').read_text())
        assert manifest['source_commit']==c['target_cache_source_commit'] and manifest['data_sha256']==c['data_sha256']
        assert sha(self.root/'targets.npz')==manifest['targets_sha256']
        with np.load(self.root/'targets.npz',allow_pickle=False) as z:
            self.ids=z['ids'].astype(str);self.y=z['labels'];self.teacher=z['teacher_scores']
            assert z['view_codes'].tolist()==list(range(8))
        root=Path(c['parent_cache']);receipt=json.loads((root/'receipt.json').read_text())
        assert receipt['source_checkpoint_sha256']==c['incumbent_sha256'] and receipt['source_commit']==os.environ['T10_DISTILL_SOURCE_COMMIT']
        assert receipt['parent_scores_sha256']==sha(root/'parent_scores.npz')
        with np.load(root/'parent_scores.npz',allow_pickle=False) as z:
            assert np.array_equal(z['ids'],self.ids) and np.array_equal(z['labels'],self.y) and z['view_codes'].tolist()==list(range(8))
            self.parent=z['scores']
        assert self.teacher.shape==self.parent.shape==(8,3496,8)
        assert np.isfinite(self.parent).all() and np.max(abs(self.parent.sum(-1)-1))<3e-6
        self.pool_mask,self.parent_correct=complement_masks(self.teacher,self.parent,self.y)
        assert self.pool_mask.sum()>=64 and len(np.unique(np.nonzero(self.pool_mask)[1]))>=16
        self.weight=1.;self.calibration_view0=False;self.bad=0;self.output=None
        self.identity=dict(teacher_cache_manifest_sha256=sha(self.root/'cache_manifest.json'),teacher_targets_sha256=manifest['targets_sha256'],
            teacher_cache_source_commit=manifest['source_commit'],
            parent_cache_receipt_sha256=sha(root/'receipt.json'),parent_checkpoint_sha256=c['incumbent_sha256'],parent_weight=.5,
            scope='KD only frozen teacher-correct/parent-wrong same-view extra4; full eight-class parent retention on all frozen-parent-correct main16',
            complement_rows=int(self.pool_mask.sum()),complement_unique_ids=len(np.unique(np.nonzero(self.pool_mask)[1])))
        if calibrated:
            p=Path(c['calibration_run'])/'receipt.json';a=json.loads(p.read_text())
            assert a['passed'] and a['profile_sha256']==sha(Path(c['_profile'])) and a['source_commit']==os.environ['T10_DISTILL_SOURCE_COMMIT']
            assert a['cache_manifest_sha256']==self.identity['teacher_cache_manifest_sha256']
            assert a['auxiliary_identity']['parent_cache_receipt_sha256']==self.identity['parent_cache_receipt_sha256']
            self.weight=a['lambda_kd'];self.identity.update(lambda_kd=self.weight,calibration_receipt_sha256=sha(p))
        # Shared calibrator expects this key; new parent is separately SHA-bound above.
        self.identity['cache_manifest_sha256']=self.identity['teacher_cache_manifest_sha256']
        self.start_round(0)

    def set_output(self,out):self.output=out

    def start_round(self,step):
        import numpy as np
        from .mixup_curriculum import sample_indices
        super().start_round(step)
        self.counts.update(extra_images=0,extra_kl_sum=0.,extra_class_counts=[0]*8)
        mask=self.pool_mask.copy()
        if self.calibration_view0:mask[1:]=False
        self.pool=np.column_stack(np.nonzero(mask));self.pool=self.pool[np.random.default_rng(17009+step).permutation(len(self.pool))]
        self.positions={int(v):j//16 for j,v in enumerate(sample_indices(self.y,step,17))} if step else {}
        if step==0:
            batches=kd.calibration_indices(self.y,self.teacher[0],self.parent[0]);excluded=set(np.concatenate(batches).tolist());self.calibration_extra={}
            for batch in batches:
                rows=pick_extra(self.pool,0,excluded);self.calibration_extra[tuple(batch.tolist())]=rows;excluded.update(index for code,index in rows)

    def start_batch(self,indices,step):
        import numpy as np
        ix=indices.numpy();codes=self.codes[ix];self.denominator=0
        self.parent_denominator=int(self.parent_correct[codes,ix].sum())
        offset=self.positions[int(ix[0])]*4 if step else 0
        rows=np.array(pick_extra(self.pool,offset,ix) if step else self.calibration_extra[tuple(ix.tolist())],dtype=np.int64)
        self.extra_codes=rows[:,0];self.extra_indices=rows[:,1]
        assert self.pool_mask[self.extra_codes,self.extra_indices].all()
        self.counts['images']+=len(ix);self.counts['protected']+=self.parent_denominator;self.counts['batches']+=1
        self.counts['extra_images']+=4
        self.counts['extra_class_counts']=(np.array(self.counts['extra_class_counts'])+np.bincount(self.y[self.extra_indices],minlength=8)).tolist()
        if self.output is not None:
            with (self.output/'auxiliary_sampling.jsonl').open('a') as f:
                f.write(json.dumps(dict(round=step,main_indices=ix.tolist(),extra_indices=self.extra_indices.tolist(),extra_ids=self.ids[self.extra_indices].tolist(),extra_codes=self.extra_codes.tolist()))+'\n')

    def raw_losses(self,probabilities,indices):
        import torch
        ix=indices.numpy();codes=self.codes[ix]
        p=torch.from_numpy(self.parent[codes,ix]).to(probabilities.device)
        mask=torch.from_numpy(self.parent_correct[codes,ix]).to(probabilities.device)
        return probabilities.sum()*0,kd.conditional_kl_part(probabilities,p,mask,self.parent_denominator)

    def loss(self,probabilities,indices,step):
        _,parent=self.raw_losses(probabilities,indices)
        self.counts['parent_conditional_sum']+=float(parent.detach())
        return .5*parent

    def extra_loss(self,model,x,step,weighted=True):
        import torch
        ix=torch.from_numpy(self.extra_indices);value=0
        for start in [0,2]:
            u=ix[start:start+2];codes=self.extra_codes[start:start+2]
            p=model(self.images_with_codes(x,u,codes))['probabilities']
            q=torch.from_numpy(self.teacher[codes,u.numpy()]).to(p.device)
            value=value+(q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum()/4
        self.counts['extra_kl_sum']+=float(value.detach())
        return self.weight*value if weighted else value

    def round_statistics(self):
        a=super().round_statistics();n=max(1,self.counts['batches'])
        a.update(auxiliary_input_images=self.counts['extra_images'],auxiliary_only_kd=True,
            weighted_complement_kd_mean=self.weight*self.counts['extra_kl_sum']/n,extra_class_counts=self.counts['extra_class_counts'])
        return a

    def diagnostics(self,model,split,step,best_correct):
        import numpy as np
        from .train import evaluate
        repair=self.pool_mask[0];retain=self.parent_correct[0];result={}
        for name,mask in [('repair',repair),('all_parent_correct',retain)]:
            ix=np.flatnonzero(mask);metrics,scores=evaluate(model,(split[0][ix],split[1][ix],split[2][ix]),4);p=scores.numpy();y=self.y[ix]
            def margin(q):
                l=np.log(q.clip(1e-12));true=l[np.arange(len(ix)),y].copy();l[np.arange(len(ix)),y]=-np.inf
                return float((true-l.max(1)).mean())
            ok=p.argmax(1)==y
            result[name]=dict(samples=len(ix),correct=int(ok.sum()),new_wrong=int((~ok).sum()) if name!='repair' else 0,
                margin=margin(p),source_margin=margin(self.parent[0,ix]),correct_per_class=np.bincount(y[ok],minlength=8).tolist(),wrong_per_class=np.bincount(y[~ok],minlength=8).tolist())
        # Compare rates because this refreshed source has a different complement.
        r=result['repair'];p=result['all_parent_correct']
        stop=step==8 and best_correct<=601 and r['correct']/r['samples']>=self.c['benchmark_repair_fraction'] and p['new_wrong']/p['samples']>=self.c['benchmark_parent_error_fraction']
        result.update(stop_for_no_transfer=stop,criterion='At8: no TEST gain and TRAIN repair fraction>=prior broadKD8, while original-correct error rate is not lower than prior broadKD8')
        return result


def prepare_parent(c,out,profile):
    import numpy as np
    import torch
    from .ensemble_teacher_cache import d4
    commit,hashes=base.sources(profile,c);model,m,protocol,splits,manifest,cache=base.prepare(c,'student');model.eval().requires_grad_(False)
    x,y,ids=splits['train'];out.mkdir(parents=True,exist_ok=False);scores=np.zeros((8,len(y),8),np.float32);codes=[]
    with torch.no_grad():
        for code in range(8):
            for start in range(0,len(y),4):
                scores[code,start:start+4]=model(d4(x[start:start+4].float().cuda()/255,code))['probabilities'].cpu().numpy()
                if start%64==0:save(out/'progress.json',dict(pid=os.getpid(),view=code,completed=start+min(4,len(y)-start),total=len(y),timestamp=time.time()))
    assert np.isfinite(scores).all() and np.max(abs(scores.sum(-1)-1))<3e-6
    np.savez_compressed(out/'parent_scores.npz',ids=ids,labels=y.numpy(),scores=scores,view_codes=np.arange(8))
    save(out/'receipt.json',dict(source_commit=commit,source_files=hashes,source_checkpoint_sha256=c['incumbent_sha256'],parent_scores_sha256=sha(out/'parent_scores.npz'),
        pid=os.getpid(),train_samples=3496,view_codes=list(range(8)),model_image_exposures=8*3496,no_val_test_forward=True,train_correct_clean=int((scores[0].argmax(1)==y.numpy()).sum())))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--recipe',choices=['student']);ap.add_argument('--prepare-parent',action='store_true');ap.add_argument('--calibrate',action='store_true');a=ap.parse_args()
    c=json.loads(a.profile.read_text());c['_profile']=str(a.profile);validate(c)
    os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']=os.environ['T10_COMPLEMENT_SOURCE_COMMIT']
    if a.prepare_parent:prepare_parent(c,a.out,a.profile);return
    if a.calibrate:kd.calibrate(c,a.out,a.profile,auxiliary_factory=Auxiliary);return
    if a.worker:base.worker(c,a.out,a.profile,'student',Auxiliary(c));return
    # Previous run must have fully exited before refreshing the parent or training.
    from .checkpoint_test_selection import ready
    root=Path(c['previous_root']);status=json.loads((root/'status.json').read_text());release=json.loads((root/'release_check.json').read_text())
    assert ready(status,release,lambda p:Path('/proc',str(p)).exists())
    assert subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['gpus']['3']
    assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=7000
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpus']['3'],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    waiting=Path(str(a.out)+'_prepare.json')
    def interrupted(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    for flag,directory in [('--prepare-parent',c['parent_cache']),('--calibrate',c['calibration_run'])]:
        out=Path(directory);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
        with Path(str(out)+'.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.complement_distill','--profile',str(a.profile),'--out',str(out),flag],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(waiting,dict(state=flag,pid=os.getpid(),child_pid=p.pid,source_commit=os.environ['T10_COMPLEMENT_SOURCE_COMMIT'],timestamp=time.time()))
            try:p.wait()
            finally:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        if p.returncode:
            save(waiting,dict(state='preparation_failed',pid=os.getpid(),phase=flag,returncode=p.returncode));raise RuntimeError('Preparation failed; no student started')
    save(waiting,dict(state='student_training',pid=os.getpid(),calibration_receipt_sha256=sha(Path(c['calibration_run'])/'receipt.json')))
    base.supervisor(c,a.out,a.profile,validator=validate)


if __name__=='__main__':main()
