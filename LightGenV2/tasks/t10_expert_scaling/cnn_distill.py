"""One admitted ImageNet-initialized TRAIN teacher, fixed T2 full49 distillation."""
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
    assert c['seed']==17 and c['rounds']==16 and c['recipes']==['student'] and c['check_rounds']==[4,8,12,16]
    assert c['batch']==16 and c['microbatch']==2 and c['samples_per_class']==128
    assert c['temperature']==2 and c['parent_weight']==.5 and c['peak_lr']==.0003 and c['end_lr']==.00003 and c['ema_decay']==.95
    assert c['sources']['student']['sha256']==c['incumbent_sha256']=='75e51f4e117f451ae7eb7947f0d001af0c07155450280fea3b19d9e9970d55a7'
    assert c['recipe_gpu']=={'student':3} and c['teacher_epochs']==8 and c['target_correct']==616


def temperature_kl_part(probabilities,teacher_logits,eligible,denominator):
    import torch.nn.functional as F
    qlog=F.log_softmax(teacher_logits/2,dim=1);plog=F.log_softmax(probabilities.clamp_min(1e-12).log()/2,dim=1)
    return (qlog.exp()*(qlog-plog)).sum(1).mul(eligible).sum()*4/max(1,int(denominator))


class Auxiliary(kd.Auxiliary):
    def __init__(self,c,calibrated=True):
        import numpy as np
        super().__init__(c,calibrated)
        with np.load(Path(c['target_cache'])/'targets.npz',allow_pickle=False) as z:self.logits=z['teacher_logits']
        assert self.logits.shape==(8,3496,8) and np.isfinite(self.logits).all()
        p=np.exp(self.logits-self.logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        assert np.max(abs(p-self.teacher))<1e-6
        self.identity.update(temperature=2,temperature_squared_factor=4,teacher_architecture='ResNet18',external_pretraining='ImageNet1K',
            teacher_logits_in_targets_sha256=self.identity['targets_sha256'],no_teacher_student_inference=True)

    def raw_losses(self,probabilities,indices):
        import torch
        ix=indices.numpy();codes=self.codes[ix];y=torch.from_numpy(self.y[ix]).to(probabilities.device)
        logits=torch.from_numpy(self.logits[codes,ix]).to(probabilities.device);p=torch.from_numpy(self.parent[codes,ix]).to(probabilities.device)
        eligible=logits.argmax(1)==y;protected=(p.argmax(1)==y)&~eligible
        return temperature_kl_part(probabilities,logits,eligible,self.denominator),kd.conditional_kl_part(probabilities,p,protected,self.parent_denominator)

    def diagnostics(self,model,split,step,best_correct):
        import numpy as np
        from .train import evaluate
        y=self.y;t=self.teacher[0].argmax(1)==y;p=self.parent[0].argmax(1)==y;result={}
        for name,mask in [('repair',t&~p),('all_parent_correct',p)]:
            ix=np.flatnonzero(mask);_,scores=evaluate(model,(split[0][ix],split[1][ix],split[2][ix]),4);scores=scores.numpy();labels=y[ix];ok=scores.argmax(1)==labels
            def margin(values):
                v=np.log(values.clip(1e-12));true=v[np.arange(len(ix)),labels].copy();v[np.arange(len(ix)),labels]=-np.inf
                return float((true-v.max(1)).mean())
            result[name]=dict(samples=len(ix),correct=int(ok.sum()),new_wrong=int((~ok).sum()) if name!='repair' else 0,
                margin=margin(scores),source_margin=margin(self.parent[0,ix]),correct_per_class=np.bincount(labels[ok],minlength=8).tolist(),wrong_per_class=np.bincount(labels[~ok],minlength=8).tolist())
        no_gain=result['repair']['margin']-result['repair']['source_margin']<=1e-5
        harm=result['all_parent_correct']['new_wrong']>=result['repair']['correct']
        if step in [4,8]:self.bad=self.bad+1 if no_gain and harm and best_correct<=601 else 0
        result.update(no_transfer_streak=self.bad,stop_for_no_transfer=self.bad>=2)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--recipe',choices=['student']);p.add_argument('--calibrate',action='store_true');a=p.parse_args();c=json.loads(a.profile.read_text());c['_profile']=str(a.profile)
    commit=os.environ['T10_CNN_SOURCE_COMMIT'];os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']=commit
    assert c['target_cache_source_commit']=='runtime_T10_CNN_SOURCE_COMMIT';c['target_cache_source_commit']=commit;validate(c)
    if a.calibrate:kd.calibrate(c,a.out,a.profile,auxiliary_factory=Auxiliary);return
    if a.worker:base.worker(c,a.out,a.profile,'student',Auxiliary(c));return
    from .checkpoint_test_selection import ready
    root=Path(c['previous_root']);assert ready(json.loads((root/'status.json').read_text()),json.loads((root/'release_check.json').read_text()),lambda pid:Path('/proc',str(pid)).exists())
    waiting=Path(str(a.out)+'_prepare.json')
    def interrupted(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def child(entry,directory,gpu,flag=None):
        out=Path(directory);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
        assert subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['gpus'][str(gpu)]
        assert int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=7000
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpus'][str(gpu)],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        cmd=[sys.executable,'-u','-m',__package__+'.'+entry,'--profile',str(a.profile),'--out',str(out)]+([flag] if flag else [])
        with Path(str(out)+'.log').open('x') as log:
            worker=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(waiting,dict(state=entry+(':'+flag if flag else ''),pid=os.getpid(),child_pid=worker.pid,source_commit=commit,timestamp=time.time()))
            try:worker.wait()
            finally:
                if worker.poll() is None:
                    os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        if worker.returncode:
            save(waiting,dict(state='preparation_failed',pid=os.getpid(),entry=entry,returncode=worker.returncode));raise RuntimeError('Preparation failed; student not launched')
    child('cnn_teacher_cache',c['target_cache'],1)
    teacher=json.loads((Path(c['target_cache'])/'status.json').read_text())
    if teacher['state']!='complete':
        save(waiting,dict(state='rejected_by_teacher_train_val_gate',pid=os.getpid(),teacher=teacher));return
    child('cnn_distill',c['calibration_run'],3,'--calibrate')
    save(waiting,dict(state='student_training',pid=os.getpid(),source_commit=commit,calibration_receipt_sha256=sha(Path(c['calibration_run'])/'receipt.json')))
    base.supervisor(c,a.out,a.profile,validator=validate)


if __name__=='__main__':main()
