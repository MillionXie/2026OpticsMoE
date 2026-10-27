"""One full49 student, frozen same-view optical ensemble targets, TRAIN calibration."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save,sha
from .ensemble_teacher_cache import d4
from . import reheat_refine as base


def validate(c):
    assert c['seed']==17 and c['recipes']==['student'] and c['rounds']==20
    assert c['check_rounds']==[4,8,12,16,20] and c['batch']==16 and c['microbatch']==2 and c['samples_per_class']==128
    assert c['schedule_type']=='distill' and c['peak_lr']==.0003 and c['end_lr']==.00003 and c['ema_decay']==.95
    assert c['recipe_gpu']=={'student':3} and c['target_correct']==616 and c['parent_weight']==.5
    assert c['temperature']==1 and c['gradient_target_ratio']==.5 and c['minimum_weighted_gradient_ratio']==.05
    assert c['sources']['student']['sha256']==c['incumbent_sha256']=='955e03e7eb8150835f231ac9288549b09ae95540b49b13c2fa8ca73f988459ee'
    assert c['gpus']=={'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}


def calibrated_weight(sup_norms,kd_norms):
    a=sum(sup_norms)/len(sup_norms);b=sum(kd_norms)/len(kd_norms)
    if not math.isfinite(a+b) or min(a,b)<=1e-12:raise ValueError('Degenerate TRAIN calibration gradient')
    value=max(.1,min(3.,.5*a/b))
    return value,value*b/a


def conditional_kl_part(probabilities,target,eligible,total_eligible):
    # The denominator is for the full effective batch, including other microbatches.
    return ((target*(target.clamp_min(1e-12).log()-probabilities.clamp_min(1e-12).log())).sum(1)*eligible).sum()/max(1,int(total_eligible))


def calibration_weight_choice(sup_norms,kd_norms,fixed=None):
    candidate,ratio=calibrated_weight(sup_norms,kd_norms)
    if fixed is None:return candidate,ratio,candidate
    if not math.isfinite(fixed) or not .1<=fixed<=3:raise ValueError('Invalid frozen control weight')
    return fixed,fixed*ratio/candidate,candidate


def calibration_indices(labels,teacher,parent):
    import numpy as np
    rng=np.random.default_rng(17);used=set();batches=[]
    t=teacher.argmax(1)==labels;p=parent.argmax(1)==labels
    for _ in range(2):
        selected=[]
        for k in range(8):
            for wanted in [t&~p,t&p]:
                choices=[int(v) for v in rng.permutation(np.flatnonzero((labels==k)&wanted)) if int(v) not in used]
                if not choices:choices=[int(v) for v in rng.permutation(np.flatnonzero(labels==k)) if int(v) not in used]
                value=choices[0];used.add(value);selected.append(value)
        batches.append(np.array(selected,dtype=np.int64))
    return batches


class Auxiliary:
    def __init__(self,c,calibrated=True):
        import numpy as np
        self.c=c;self.root=Path(c['target_cache']);mp=self.root/'cache_manifest.json';manifest=json.loads(mp.read_text())
        assert manifest['source_commit']==c['target_cache_source_commit']
        assert sha(self.root/'targets.npz')==manifest['targets_sha256'] and manifest['data_sha256']==c['data_sha256']
        assert manifest['configuration']['models']['parent']['sha256']==c['incumbent_sha256']
        with np.load(self.root/'targets.npz',allow_pickle=False) as z:
            self.ids=z['ids'].astype(str);self.y=z['labels'];self.teacher=z['teacher_scores'];self.parent=z['parent_scores'];assert z['view_codes'].tolist()==list(range(8))
        assert self.teacher.shape==self.parent.shape==(8,3496,8) and np.isfinite(self.teacher).all() and np.isfinite(self.parent).all()
        assert max(np.abs(self.teacher.sum(-1)-1).max(),np.abs(self.parent.sum(-1)-1).max())<3e-6
        self.weight=1.;self.calibration_view0=False;self.bad=0
        self.identity=dict(cache_manifest_sha256=sha(mp),targets_sha256=manifest['targets_sha256'],cache_source_commit=manifest['source_commit'],
            teacher_weights=manifest['teacher_weights'],temperature=1,parent_weight=.5,conditional_denominator='Whole effective16 batch')
        if calibrated:
            receipt=Path(c['calibration_run'])/'receipt.json';q=json.loads(receipt.read_text())
            assert q['passed'] and q['profile_sha256']==sha(Path(c['_profile'])) and q['source_commit']==os.environ['T10_DISTILL_SOURCE_COMMIT']
            assert q['cache_manifest_sha256']==self.identity['cache_manifest_sha256']
            self.weight=q['lambda_kd'];self.identity.update(lambda_kd=self.weight,calibration_receipt_sha256=sha(receipt))
        self.codes=None;self.start_round(0)

    def configure(self,splits):
        import numpy as np
        assert np.array_equal(splits['train'][2],self.ids) and np.array_equal(splits['train'][1].numpy(),self.y)
        assert set(self.ids).isdisjoint(splits['val'][2])

    def start_round(self,step):
        import numpy as np
        self.codes=np.zeros(3496,dtype=np.int64) if self.calibration_view0 else np.random.default_rng(17+step*1000033).integers(0,8,3496)
        self.counts=dict(images=0,eligible=0,protected=0,kd_conditional_sum=0.,parent_conditional_sum=0.,batches=0)

    def start_batch(self,indices,step):
        ix=indices.numpy();codes=self.codes[ix];y=self.y[ix]
        e=self.teacher[codes,ix].argmax(1)==y;p=(self.parent[codes,ix].argmax(1)==y)&~e
        self.denominator=int(e.sum());self.parent_denominator=int(p.sum())
        self.counts['images']+=len(ix);self.counts['eligible']+=self.denominator;self.counts['protected']+=self.parent_denominator;self.counts['batches']+=1

    def images_with_codes(self,x,indices,codes):
        import numpy as np
        import torch
        batch=x[indices].float().cuda()/255
        for code in range(8):
            ix=np.flatnonzero(codes==code)
            if len(ix):
                selection=torch.from_numpy(ix).to(batch.device);batch[selection]=d4(batch[selection],code)
        return batch

    def images(self,x,indices,step):return self.images_with_codes(x,indices,self.codes[indices.numpy()])

    def raw_losses(self,probabilities,indices):
        import torch
        ix=indices.numpy();codes=self.codes[ix];y=self.y[ix]
        q=torch.from_numpy(self.teacher[codes,ix]).to(probabilities.device);p=torch.from_numpy(self.parent[codes,ix]).to(probabilities.device)
        e=q.argmax(1)==torch.from_numpy(y).to(probabilities.device);protect=(p.argmax(1)==torch.from_numpy(y).to(probabilities.device))&~e
        return conditional_kl_part(probabilities,q,e,self.denominator),conditional_kl_part(probabilities,p,protect,self.parent_denominator)

    def loss(self,probabilities,indices,step):
        kd,parent=self.raw_losses(probabilities,indices)
        self.counts['kd_conditional_sum']+=float(kd.detach());self.counts['parent_conditional_sum']+=float(parent.detach())
        return self.weight*kd+.5*parent

    def round_statistics(self):
        n=max(1,self.counts['batches']);total=max(1,self.counts['images'])
        return dict(lambda_kd=self.weight,teacher_eligible_fraction=self.counts['eligible']/total,parent_protected_fraction=self.counts['protected']/total,
            kd_conditional_mean=self.counts['kd_conditional_sum']/n,weighted_kd_mean=self.weight*self.counts['kd_conditional_sum']/n,
            weighted_parent_mean=.5*self.counts['parent_conditional_sum']/n,optimizer_batches=self.counts['batches'])

    def diagnostics(self,model,split,step,best_correct):
        import numpy as np
        from .train import evaluate
        y=self.y;t=self.teacher[0].argmax(1)==y;p=self.parent[0].argmax(1)==y;result={}
        for name,mask in [('repair',t&~p),('protected',p&~t)]:
            ix=np.flatnonzero(mask)
            assert len(ix)>0
            selected=(split[0][ix],split[1][ix],split[2][ix]);metrics,scores=evaluate(model,selected,4);scores=scores.numpy();labels=y[ix]
            def margin(values):
                values=np.log(values.clip(1e-12));correct=values[np.arange(len(ix)),labels];wrong=values.copy();wrong[np.arange(len(ix)),labels]=-np.inf
                return float((correct-wrong.max(1)).mean())
            result[name]=dict(samples=len(ix),correct=int((scores.argmax(1)==labels).sum()),new_wrong=int((scores.argmax(1)!=labels).sum()) if name=='protected' else 0,
                margin=margin(scores),source_margin=margin(self.parent[0,ix]))
        no_gain=result['repair']['margin']-result['repair']['source_margin']<=1e-5
        harm=result['protected']['new_wrong']>=result['repair']['correct']
        if step in [4,8]:self.bad=self.bad+1 if no_gain and harm and best_correct<=601 else 0
        result.update(no_transfer_streak=self.bad,stop_for_no_transfer=self.bad>=2)
        return result


def calibrate(c,out,profile,auxiliary_factory=Auxiliary):
    import numpy as np
    import torch
    from .train import objective
    from .diagnose_topk import classification_margin_loss
    from .sharpness_refine import optimization_step
    commit,hashes=base.sources(profile,c);aux=auxiliary_factory(c,False);aux.calibration_view0=True;aux.start_round(0)
    model,m,protocol,splits,manifest,cache=base.prepare(c,'student');aux.configure(splits);x,y,ids=splits['train'];tc=protocol['training_proposal']
    original={n:p.detach().cpu().clone() for n,p in model.state_dict().items()};weights=len(y)/(8*torch.bincount(y,minlength=8).float().cuda())
    batches=aux.calibration_batches() if hasattr(aux,'calibration_batches') else calibration_indices(aux.y,aux.teacher[0],aux.parent[0]);norms=[];rows=[]
    def vector():return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    def norm(v):return float(torch.stack([p.norm() for p in v]).norm())
    def relationship(a,b):
        na=norm(a);nb=norm(b)
        return dict(supervised_norm=na,kd_norm=nb,cosine=float(torch.stack([(p*q).sum() for p,q in zip(a,b)]).sum())/max(na*nb,1e-30))
    def gradient(ix,kind):
        model.zero_grad(set_to_none=True);model.train();aux.start_batch(ix,0)
        if kind=='kd' and hasattr(aux,'extra_loss'):
            aux.extra_loss(model,x,0,weighted=False).backward()
            return vector()
        for start in range(0,len(ix),2):
            u=ix[start:start+2];o=model(aux.images(x,u,0));labels=y[u].cuda()
            if kind=='sup':
                lp=o['probabilities'].clamp_min(1e-12).log();loss=-((1-.02)*lp[torch.arange(len(u),device=lp.device),labels]+.02*lp.mean(1)).mean()+.2*classification_margin_loss(o['probabilities'],labels);loss=loss*len(u)/len(ix)
            else:loss=aux.raw_losses(o['probabilities'],u)[0]
            loss.backward()
        return vector()
    for no,values in enumerate(batches,1):
        ix=torch.from_numpy(values);gs=gradient(ix,'sup');gk=gradient(ix,'kd');r=relationship(gs,gk);norms.append((r['supervised_norm'],r['kd_norm']));del gs,gk
        groups={}
        for group,mask in ([] if hasattr(aux,'extra_loss') else [('repair',(aux.teacher[0].argmax(1)==aux.y)&(aux.parent[0].argmax(1)!=aux.y)),('already_correct',aux.parent[0].argmax(1)==aux.y)]):
            selected=values[mask[values]]
            if len(selected):
                a=gradient(torch.from_numpy(selected),'sup');b=gradient(torch.from_numpy(selected),'kd');groups[group]=dict(samples=len(selected),**relationship(a,b));del a,b
        rows.append(dict(batch=no,indices=values.tolist(),ids=ids[values].tolist(),classes=np.bincount(aux.y[values],minlength=8).tolist(),gradients=r,groups=groups))
        if hasattr(aux,'extra_loss'):
            rows[-1].update(extra_indices=aux.extra_indices.tolist(),extra_codes=aux.extra_codes.tolist(),extra_ids=ids[aux.extra_indices].tolist())
    weight,weighted_ratio,candidate_weight=calibration_weight_choice([p[0] for p in norms],[p[1] for p in norms],c.get('fixed_lambda_kd'));aux.weight=weight
    # Verify all eight TRAIN encodings against the frozen parent cache before any optimizer probe.
    equivalence=[];ix=torch.from_numpy(batches[0][:2]);model.eval()
    with torch.no_grad():
        for code in range(8):
            p=model(aux.images_with_codes(x,ix,np.full(len(ix),code)))['probabilities'].cpu().numpy();err=float(np.abs(p-aux.parent[code,ix.numpy()]).max());assert err<=1e-6
            equivalence.append(err)
    for row,values in zip(rows,batches):
        ix=torch.from_numpy(values);updates=[];step_info=[]
        for mode in ['baseline','repeat_baseline','distill']:
            model.load_state_dict(original);model.train();aux.start_round(0);aux.start_batch(ix,0)
            opt=torch.optim.AdamW([{'params':list(model.expert_phases),'lr':.0001},{'params':list(model.global_phases),'lr':.00005},{'params':[model.router_phase],'lr':.00001}],weight_decay=0.)
            def closure():
                value=0.
                for start in range(0,16,2):
                    u=ix[start:start+2];output=model(aux.images(x,u,0));loss=(objective(model,output,y[u].cuda(),weights,tc)+.2*classification_margin_loss(output['probabilities'],y[u].cuda()))/8
                    if mode=='distill':loss=loss+aux.loss(output['probabilities'],u,0)
                    loss.backward();value+=float(loss.detach())
                if mode=='distill' and hasattr(aux,'extra_loss'):
                    extra=aux.extra_loss(model,x,0);extra.backward();value+=float(extra.detach())
                return value
            info=optimization_step(model,opt,closure);assert len(info['gradients'])==5 and all(v['finite'] and v['norm']>0 for v in info['gradients'].values())
            change=[(p.detach().cpu()-original[n]).flatten() for n,p in model.named_parameters()]
            assert all(torch.isfinite(v).all() and float(v.norm())>0 for v in change)
            info['parameter_update_norms']={n:float(v.norm()) for (n,p),v in zip(model.named_parameters(),change)}
            updates.append(change);step_info.append(info)
        first=torch.cat(updates[0]);repeat=torch.cat(updates[1]);changed=torch.cat(updates[2]);denom=float(first.norm());repeat_error=float((repeat-first).norm())/denom;relative=float((changed-first).norm())/denom
        row['actual_adam_update']=dict(cosine=float((first*changed).sum())/max(float(first.norm()*changed.norm()),1e-30),relative_change=relative,repeat_relative_error=repeat_error,finite_all_groups=True,steps=step_info)
    model.load_state_dict(original);assert all(torch.equal(p.detach().cpu(),original[n]) for n,p in model.state_dict().items())
    passed=weighted_ratio>=c['minimum_weighted_gradient_ratio'] and all(r['actual_adam_update']['relative_change']>10*r['actual_adam_update']['repeat_relative_error']+1e-7 for r in rows)
    out.mkdir(parents=True,exist_ok=False);save(out/'receipt.json',dict(source_commit=commit,source_files=hashes,profile_sha256=sha(profile),pid=os.getpid(),
        cache_manifest_sha256=aux.identity['cache_manifest_sha256'],lambda_kd=weight,weighted_gradient_ratio=weighted_ratio,
        calibration_target_ratio=.5,lambda_bounds=[.1,3.],batches=rows,unique_train_samples=32,optimizer_probe_updates=6,
        weight_frozen_from_matched_control='fixed_lambda_kd' in c,lambda_recalibration_candidate=candidate_weight,
        source_restored_exactly=True,parent_cache_d4_max_errors=equivalence,auxiliary_identity=aux.identity,
        no_val_test_forward=not c.get('test_images_enter_student_gradient',False),
        calibration_scope='TRAIN+TEST' if c.get('test_images_enter_student_gradient',False) else 'TRAIN',passed=passed))
    if not passed:raise RuntimeError('TRAIN numeric KD admission rejected')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--recipe',choices=['student']);ap.add_argument('--calibrate',action='store_true');a=ap.parse_args();c=json.loads(a.profile.read_text());c['_profile']=str(a.profile);validate(c)
    os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']
    if a.calibrate:calibrate(c,a.out,a.profile);return
    if a.worker:base.worker(c,a.out,a.profile,'student',Auxiliary(c));return
    # CPU-only bounded waiter. Do not import Torch or compete with either producer.
    from .checkpoint_test_selection import ready
    cache=Path(c['target_cache']);started=time.time();waiting=Path(str(a.out)+'_wait.json')
    while True:
        status=json.loads((cache/'status.json').read_text());rp=cache/'release_check.json';release=json.loads(rp.read_text()) if rp.exists() else {}
        if status['state'] in ['failed','rejected_by_train_audit']:
            save(waiting,dict(state='rejected_before_training',pid=os.getpid(),cache_status=status));return
        if ready(status,release,lambda p:Path('/proc',str(p)).exists()):break
        if time.time()-started>7200:raise TimeoutError('Cache wait expired without interrupting producers')
        save(waiting,dict(state='waiting_for_cache_release',pid=os.getpid(),cache_status=status,source_commit=os.environ['T10_DISTILL_SOURCE_COMMIT'],timestamp=time.time()));time.sleep(20)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpus']['3'],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    calibration=Path(c['calibration_run']);assert not calibration.exists();log=Path(str(calibration)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    save(waiting,dict(state='calibrating_train_gradients',pid=os.getpid(),source_commit=os.environ['T10_DISTILL_SOURCE_COMMIT'],timestamp=time.time()))
    def interrupted(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with log.open('x') as stream:
        process=subprocess.Popen([sys.executable,'-u','-m',__package__+'.cached_distill','--profile',str(a.profile),'--out',str(calibration),'--calibrate'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save(waiting,dict(state='calibrating_train_gradients',pid=os.getpid(),calibration_pid=process.pid,source_commit=os.environ['T10_DISTILL_SOURCE_COMMIT'],timestamp=time.time()))
        try:process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    if process.returncode:
        save(waiting,dict(state='calibration_failed',pid=os.getpid(),calibration_log=str(log)));raise RuntimeError('TRAIN calibration failed; no student started')
    save(waiting,dict(state='student_training',pid=os.getpid(),calibration_receipt_sha256=sha(calibration/'receipt.json'),timestamp=time.time()))
    base.supervisor(c,a.out,a.profile,validator=validate)


if __name__=='__main__':main()
