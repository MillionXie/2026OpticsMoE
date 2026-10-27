"""Explicitly TEST-exposed teacher; matched TRAIN-only / TRAIN+TEST students.

All TEST numbers here are training-exposed development fit scores. This does not
establish a held-out improvement over the original Top-k experiment.
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
from .cnn_distill import temperature_kl_part
from .cnn_teacher_cache import inputs,codes_for_epoch
from .ensemble_teacher_cache import d4
from .expert_tail_schedule import save,sha


def validate(c):
    assert c['explicit_test_training_authorization'] is True
    assert c['recipes']==['train_only','train_test'] and c['rounds']==16
    assert c['check_rounds']==[4,8,12,16] and c['batch']==16 and c['microbatch']==2
    assert c['teacher_epochs']==8 and c['temperature']==2 and c['parent_weight']==.5
    assert c['recipe_gpu']=={'train_only':1,'train_test':3}
    assert c['peak_lr']==.0003 and c['end_lr']==.00003 and c['seed']==17
    assert c['incumbent_sha256']=='f6d718bd04ab9f2d99ff759fadc4654d82d04454b7c7b2df9bac4300ca97128c'


def load_pool(c):
    import numpy as np
    p=Path(c['teacher_data']);assert sha(p)==c['data_sha256']
    with np.load(p,allow_pickle=False) as z:
        rows=[(z[s+'_images'],z[s+'_labels'].astype(np.int64),z[s+'_ids'].astype(str)) for s in ['train','test','val']]
    tr,te,va=rows
    assert [len(v[1]) for v in rows]==[3496,752,752]
    assert len(set(tr[2])|set(te[2])|set(va[2]))==5000
    x=np.concatenate([tr[0],te[0]]);y=np.concatenate([tr[1],te[1]]);ids=np.concatenate([tr[2],te[2]])
    assert np.bincount(y,minlength=8).tolist()==[531]*8
    return x,y,ids,va


def teacher_gate(train_correct,test_correct,val_correct):
    return train_correct>=3427 and test_correct>=737 and val_correct>=715


def teacher(c,out,profile):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18
    commit,hashes=base.sources(profile,c);x,y,ids,va=load_pool(c)
    source=Path(c['teacher_source']);assert sha(source)==c['teacher_source_sha256']
    torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(4)
    model=resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,8)
    model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['model']);model=model.cuda()
    opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.startswith('fc.')],'initial_lr':5e-5},
        {'params':model.fc.parameters(),'initial_lr':2e-4}],weight_decay=1e-4)
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    save(out/'metadata.json',dict(pid=os.getpid(),source_commit=commit,source_files=hashes,configuration=c,
        teacher_source_sha256=sha(source),gradient_splits=['train','test'],validation_split='val',
        test_metric_role='Training-exposed fit score; not held-out',augmentation='Exact D4, each image every view once in8epochs'))
    def predict(xx,yy,ii,code=0):
        model.eval();logits=np.zeros((len(yy),8),np.float32)
        with torch.no_grad():
            for start in range(0,len(yy),64):
                ix=np.arange(start,min(start+64,len(yy)));logits[ix]=model(inputs(xx,ix,np.full(len(ix),code))).cpu().numpy()
        scores=np.exp(logits-logits.max(1,keepdims=True));scores/=scores.sum(1,keepdims=True)
        assert np.isfinite(scores).all()
        ok=scores.argmax(1)==yy
        return logits,scores,dict(correct=int(ok.sum()),total=len(yy),accuracy=float(ok.mean()),macro_nll=float(-np.log(scores[np.arange(len(yy)),yy].clip(1e-12)).mean()),correct_per_class=np.bincount(yy[ok],minlength=8).tolist())
    history=[];best=(-1,float('-inf'));selected=None
    for epoch in range(1,9):
        model.train();order=np.random.default_rng(17+epoch).permutation(len(y));codes=codes_for_epoch(len(y),epoch)
        with (out/'sampling_history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,indices=order.tolist(),ids=ids[order].tolist(),codes=codes[order].tolist()))+'\n')
        scale=.1+.9*(1+np.cos(np.pi*(epoch-1)/7))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*scale
        loss_sum=0.;t0=time.time()
        for start in range(0,len(y),32):
            ix=order[start:start+32];opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(inputs(x,ix,codes[ix])),torch.from_numpy(y[ix]).cuda())
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
            opt.step();loss_sum+=float(loss.detach())*len(ix)
            if start%256==0:save(out/'status.json',dict(state='training_test_exposed_teacher',pid=os.getpid(),epoch=epoch,completed=start+len(ix),total=len(y),timestamp=time.time()))
        _,test_scores,tm=predict(x[3496:],y[3496:],ids[3496:]);_,val_scores,vm=predict(*va)
        row=dict(epoch=epoch,loss=loss_sum/len(y),test_fit=tm,val=vm,training_seconds=time.time()-t0,gradient_norm=float(norm));history.append(row)
        torch.save(dict(model=model.state_dict(),epoch=epoch,optimizer=opt.state_dict(),source_commit=commit),out/'last_checkpoint.pt')
        # TEST is now an explicitly authorized training split. Enforce a VAL floor,
        # then select its fit count/NLL; never present this as validation on TEST.
        key=(tm['correct'],-tm['macro_nll']) if vm['correct']>=715 else (-1,float('-inf'))
        if key>best:
            best=key;selected=row;torch.save(dict(model=model.state_dict(),epoch=epoch,source_commit=commit),out/'best_checkpoint.pt')
            np.savez_compressed(out/'selected_test_fit_predictions.npz',ids=ids[3496:],labels=y[3496:],scores=test_scores)
            np.savez_compressed(out/'selected_val_predictions.npz',ids=va[2],labels=va[1],scores=val_scores)
        save(out/'history.json',history)
    if selected is None:raise RuntimeError('No teacher candidate met fixed VAL floor715')
    model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'])
    logits,scores,pm=predict(x,y,ids);train_correct=int((scores[:3496].argmax(1)==y[:3496]).sum())
    ok=teacher_gate(train_correct,selected['test_fit']['correct'],selected['val']['correct'])
    np.savez_compressed(out/'clean_pool_predictions.npz',ids=ids,labels=y,scores=scores,logits=logits)
    admission=dict(admitted=ok,train_correct=train_correct,test_fit=selected['test_fit'],val=selected['val'],selected_epoch=selected['epoch'],
        thresholds=dict(train_correct=3427,test_fit_correct=737,val_correct=715),test_exposed_training=True,teacher_checkpoint_sha256=sha(out/'best_checkpoint.pt'))
    save(out/'admission.json',admission)
    if not ok:save(out/'status.json',dict(state='rejected_by_fixed_teacher_gate',pid=os.getpid(),admission=admission));return
    all_logits=np.zeros((8,len(y),8),np.float32);all_scores=np.zeros_like(all_logits);all_logits[0]=logits;all_scores[0]=scores
    for code in range(1,8):
        all_logits[code],all_scores[code],_=predict(x,y,ids,code)
        save(out/'status.json',dict(state='caching_test_exposed_teacher',pid=os.getpid(),view=code,completed=len(y),timestamp=time.time()))
    np.savez_compressed(out/'teacher_targets.npz',ids=ids,labels=y,teacher_logits=all_logits,teacher_scores=all_scores,view_codes=np.arange(8),split_origin=np.array(['train']*3496+['test']*752))
    save(out/'teacher_manifest.json',dict(source_commit=commit,source_files=hashes,data_sha256=c['data_sha256'],targets_sha256=sha(out/'teacher_targets.npz'),teacher_checkpoint_sha256=sha(out/'best_checkpoint.pt'),
        gradient_splits=['train','test'],train_inputs=8*4248,val_forward_inputs=8*752,test_fit_selection_forward_inputs=8*752,cache_forward_inputs=8*4248,elapsed_seconds=time.time()-started))
    save(out/'status.json',dict(state='complete',pid=os.getpid(),admitted=True,train_correct=train_correct,test_fit_correct=selected['test_fit']['correct'],val_correct=selected['val']['correct']))


def parent(c,out,profile):
    import numpy as np
    import torch
    commit,hashes=base.sources(profile,c);model,_,_,_,_,_=base.prepare(c,'student');model.eval().requires_grad_(False)
    x,y,ids,_=load_pool(c);x=torch.from_numpy(x.transpose(0,3,1,2).copy());out.mkdir(parents=True,exist_ok=False)
    scores=np.zeros((8,len(y),8),np.float32)
    with torch.no_grad():
        for code in range(8):
            for start in range(0,len(y),4):
                scores[code,start:start+4]=model(d4(x[start:start+4].float().cuda()/255,code))['probabilities'].cpu().numpy()
                if start%64==0:save(out/'progress.json',dict(pid=os.getpid(),view=code,completed=min(start+4,len(y)),total=len(y),timestamp=time.time()))
    assert np.isfinite(scores).all() and np.max(abs(scores.sum(-1)-1))<3e-6
    np.savez_compressed(out/'parent_scores.npz',ids=ids,labels=y,scores=scores,view_codes=np.arange(8))
    save(out/'receipt.json',dict(source_commit=commit,source_files=hashes,pid=os.getpid(),source_checkpoint_sha256=c['incumbent_sha256'],data_sha256=c['data_sha256'],
        parent_scores_sha256=sha(out/'parent_scores.npz'),model_image_inputs=8*4248,gradient_updates=0,forward_splits=['train','test']))


def scope_config(c,recipe):
    v=copy.deepcopy(c);v['test_images_enter_student_gradient']=recipe=='train_test';v['calibration_run']=c['calibration_runs'][recipe]
    return v


class Auxiliary(kd.Auxiliary):
    def __init__(self,c,calibrated=True):
        import numpy as np
        self.c=c;self.root=Path(c['target_cache']);m=json.loads((self.root/'teacher_manifest.json').read_text());r=Path(c['parent_cache']);pr=json.loads((r/'receipt.json').read_text())
        commit=os.environ['T10_DISTILL_SOURCE_COMMIT']
        assert m['source_commit']==c.get('teacher_cache_source_commit',commit) and pr['source_commit']==c.get('parent_cache_source_commit',commit)
        parent_sha=c.get('parent_checkpoint_sha256',c['incumbent_sha256'])
        assert m['data_sha256']==pr['data_sha256']==c['data_sha256'] and pr['source_checkpoint_sha256']==parent_sha
        assert sha(self.root/'teacher_targets.npz')==m['targets_sha256'] and sha(r/'parent_scores.npz')==pr['parent_scores_sha256']
        n=4248 if c['test_images_enter_student_gradient'] else 3496
        with np.load(self.root/'teacher_targets.npz',allow_pickle=False) as z,np.load(r/'parent_scores.npz',allow_pickle=False) as p:
            assert np.array_equal(z['ids'],p['ids']) and np.array_equal(z['labels'],p['labels'])
            assert z['view_codes'].tolist()==p['view_codes'].tolist()==list(range(8))
            self.ids=z['ids'][:n].astype(str);self.y=z['labels'][:n];self.teacher=z['teacher_scores'][:,:n];self.logits=z['teacher_logits'][:,:n];self.parent=p['scores'][:,:n]
        assert self.teacher.shape==self.logits.shape==self.parent.shape==(8,n,8)
        q=np.exp(self.logits-self.logits.max(-1,keepdims=True));q/=q.sum(-1,keepdims=True)
        assert np.max(abs(q-self.teacher))<1e-6 and np.isfinite(self.parent).all() and np.max(abs(self.parent.sum(-1)-1))<3e-6
        self.weight=1.;self.calibration_view0=False;self.bad=0
        self.identity=dict(cache_manifest_sha256=sha(self.root/'teacher_manifest.json'),parent_receipt_sha256=sha(r/'receipt.json'),targets_sha256=m['targets_sha256'],
            teacher_test_training=True,student_gradient_splits=['train','test'] if n==4248 else ['train'],samples=n,temperature=2,external_pretraining='ImageNet1K',parent_checkpoint_sha256=parent_sha,
            teacher_cache_source_commit=m['source_commit'],parent_cache_source_commit=pr['source_commit'])
        if calibrated:
            p=Path(c['calibration_run'])/'receipt.json';a=json.loads(p.read_text())
            assert a['passed'] and a['profile_sha256']==sha(Path(c['_profile'])) and a['source_commit']==commit
            assert a['auxiliary_identity']['student_gradient_splits']==self.identity['student_gradient_splits']
            assert a['cache_manifest_sha256']==self.identity['cache_manifest_sha256'] and a['auxiliary_identity']['parent_receipt_sha256']==self.identity['parent_receipt_sha256']
            self.weight=a['lambda_kd'];self.identity.update(lambda_kd=self.weight,calibration_receipt_sha256=sha(p))
        self.start_round(0)

    def configure(self,splits):
        import numpy as np
        import torch
        if self.c['test_images_enter_student_gradient']:
            x,y,ids,_=load_pool(self.c);splits['original_train']=splits['train']
            splits['train']=(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y),ids)
        assert np.array_equal(splits['train'][2],self.ids) and np.array_equal(splits['train'][1].numpy(),self.y)
        assert set(self.ids).isdisjoint(splits['val'][2])

    def start_round(self,step):
        import numpy as np
        self.codes=np.zeros(len(self.y),np.int64) if self.calibration_view0 else np.random.default_rng(17+step*1000033).integers(0,8,len(self.y))
        self.counts=dict(images=0,eligible=0,protected=0,kd_conditional_sum=0.,parent_conditional_sum=0.,batches=0)

    def raw_losses(self,probabilities,indices):
        import torch
        ix=indices.numpy();codes=self.codes[ix];y=torch.from_numpy(self.y[ix]).to(probabilities.device)
        logits=torch.from_numpy(self.logits[codes,ix]).to(probabilities.device);p=torch.from_numpy(self.parent[codes,ix]).to(probabilities.device)
        eligible=logits.argmax(1)==y;protected=(p.argmax(1)==y)&~eligible
        return temperature_kl_part(probabilities,logits,eligible,self.denominator),kd.conditional_kl_part(probabilities,p,protected,self.parent_denominator)

    def diagnostics(self,model,split,step,best_correct):
        # This fixed comparison tests two exposure scopes. No extra adaptive
        # transfer/TEST-triggered stopping is added to its sixteen-round budget.
        return dict(stop_for_no_transfer=False,teacher_test_exposed=True,student_gradient_splits=self.identity['student_gradient_splits'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--teacher',action='store_true');p.add_argument('--parent',action='store_true');p.add_argument('--calibrate',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--recipe',choices=['train_only','train_test'])
    a=p.parse_args();c=json.loads(a.profile.read_text());c['_profile']=str(a.profile);validate(c)
    commit=os.environ['T10_TEST_EXPOSED_SOURCE_COMMIT'];os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']=commit
    if a.teacher:teacher(c,a.out,a.profile);return
    if a.parent:parent(c,a.out,a.profile);return
    if a.calibrate or a.worker:
        c=scope_config(c,a.recipe)
        if a.calibrate:
            kd.calibrate(c,a.out,a.profile,auxiliary_factory=Auxiliary)
            r=json.loads((a.out/'receipt.json').read_text());_,_,pool_ids,_=load_pool(c);origins={str(v):'train' if j<3496 else 'test' for j,v in enumerate(pool_ids)}
            for b in r['batches']:b['split_origins']=[origins[str(v)] for v in b['ids']]
            save(a.out/'receipt.json',r)
        else:base.worker(c,a.out,a.profile,a.recipe,Auxiliary(c))
        return
    waiting=Path(str(a.out)+'_prepare.json');running={}
    def interrupt(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def launch(out,index,flags):
        out=Path(out);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
        assert subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['gpus'][str(index)]
        assert int(subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpus'][str(index)],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        log=Path(str(out)+'.log').open('x');cmd=[sys.executable,'-u','-m',__package__+'.test_exposed_distill','--profile',str(a.profile),'--out',str(out),*flags]
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[child.pid]=(child,log)
        return child.pid
    def join(stage):
        while running:
            save(waiting,dict(state=stage,pid=os.getpid(),running=list(running),source_commit=commit,timestamp=time.time()))
            for pid,(child,log) in list(running.items()):
                if child.poll() is None:continue
                log.close();del running[pid]
                if child.returncode:raise RuntimeError('Preparation failed '+str(pid)+'; logs preserved')
            if running:time.sleep(10)
    try:
        base.sources(a.profile,c)
        launch(c['target_cache'],1,['--teacher']);launch(c['parent_cache'],3,['--parent']);join('teacher_and_parent_cache')
        if not json.loads((Path(c['target_cache'])/'admission.json').read_text())['admitted']:
            save(waiting,dict(state='teacher_rejected',pid=os.getpid()));return
        for recipe in c['recipes']:launch(c['calibration_runs'][recipe],c['recipe_gpu'][recipe],['--calibrate','--recipe',recipe])
        join('calibrating_both_scopes')
        save(waiting,dict(state='student_training',pid=os.getpid(),source_commit=commit))
        base.supervisor(c,a.out,a.profile,validator=validate)
    except BaseException as e:
        save(waiting,dict(state='failed',pid=os.getpid(),error=repr(e)));raise
    finally:
        for child,log in running.values():
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            log.close()


if __name__=='__main__':main()
