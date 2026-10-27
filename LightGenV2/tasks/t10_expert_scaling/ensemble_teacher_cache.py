"""TRAIN-only admission audit and exact-view cache for three frozen optical teachers.

No optimizer, VAL/TEST forward or student training is performed here. The clean
TRAIN gate is fixed before inference. Only an admitted run computes the other
seven D4 views, using one producer on each authorized GPU.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha


def admission(labels, teacher, parent, minimum=64, classes=4, per_class=4):
    import numpy as np
    y=np.asarray(labels);t=teacher.argmax(1)==y;p=parent.argmax(1)==y
    complement=t&~p;counts=np.bincount(y[complement],minlength=8)
    return dict(teacher_correct=int(t.sum()),parent_correct=int(p.sum()),
        teacher_correct_parent_wrong=int(complement.sum()),parent_correct_teacher_wrong=int((p&~t).sum()),
        common_wrong=int((~p&~t).sum()),complement_per_class=counts.tolist(),
        teacher_macro_nll=float(np.mean([-np.log(teacher[y==k,k].clip(1e-12)).mean() for k in range(8)])),
        parent_macro_nll=float(np.mean([-np.log(parent[y==k,k].clip(1e-12)).mean() for k in range(8)])),
        teacher_mean_true_probability=float(teacher[np.arange(len(y)),y].mean()),
        parent_mean_true_probability=float(parent[np.arange(len(y)),y].mean()),
        admitted=bool(complement.sum()>=minimum and (counts>=per_class).sum()>=classes),
        gate=dict(minimum_complement=minimum,minimum_classes=classes,minimum_per_class=per_class))


def d4(batch, code):
    """Eight unique views: rotate0..3, then optional horizontal reflection."""
    if not 0<=code<8:raise ValueError('D4 code outside0..7')
    result=batch.rot90(code%4,(-2,-1))
    return result.flip(-1) if code//4 else result


def sources(profile,c):
    commit=os.environ['T10_TEACHER_CACHE_SOURCE_COMMIT'];repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    paths=[Path(__file__),profile,*[Path(__file__).with_name(n) for n in ['model.py','train.py','plan.py','expert_tail_schedule.py']],
        Path(__file__).parent/'configs/study.json',repo/'LightGenV2/common/optical_contract.py'];hashes={}
    for p in paths:
        name=p.resolve().relative_to(repo).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+name]);hashes[name]=sha(p)
    for row in c['models'].values():assert sha(Path(row['run'])/'best_checkpoint.pt')==row['sha256']
    return commit,hashes


def produce(c,out,profile,tags,mode):
    import numpy as np
    import torch
    from .model import ScalingOptics
    commit,hashes=sources(profile,c);torch.set_num_threads(4)
    cache=Path(c['data']);manifest=json.loads((cache.parent/'data_manifest.json').read_text());assert sha(cache)==c['data_sha256']==manifest['cache_sha256']
    with np.load(cache,allow_pickle=False) as z:x=z['train_images'];y=z['train_labels'];ids=z['train_ids'].astype(str)
    assert len(ids)==3496 and ids.tolist()==manifest['split_ids']['train']
    assert set(ids).isdisjoint(manifest['split_ids']['val']) and set(ids).isdisjoint(manifest['split_ids']['test'])
    images=torch.from_numpy(x.transpose(0,3,1,2).copy());last_progress=0.
    for tag in tags:
        row=c['models'][tag];src=Path(row['run']);m=json.loads((src/'metadata.json').read_text());a=m['arguments']
        assert a['experts']==49 and a['top_k']==row['top_k'] and a['layers']==4 and m['data_sha256']==c['data_sha256']
        model=ScalingOptics(m['protocol'],49,row['top_k'],'moe_oeo',8,a['layers'],a['padding']).cuda().eval().requires_grad_(False)
        checkpoint=torch.load(src/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);counter=checkpoint['epoch'];del checkpoint
        dest=out/tag;dest.mkdir(parents=True,exist_ok=True);scores=np.empty((8,len(y),8),dtype=np.float32)
        if mode=='full':
            assert sha(dest/'clean_scores.npz')==json.loads((dest/'clean_receipt.json').read_text())['scores_sha256']
            with np.load(dest/'clean_scores.npz',allow_pickle=False) as z:
                assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],y);scores[0]=z['scores']
        codes=[0] if mode=='clean' else list(range(1,8));started=time.time()
        with torch.no_grad():
            for code in codes:
                for start in range(0,len(y),4):
                    ix=torch.arange(start,min(start+4,len(y)));batch=d4(images[ix].float().cuda()/255,code)
                    result=model(batch)['probabilities'];assert torch.isfinite(result).all() and torch.max(abs(result.sum(1)-1))<3e-6;scores[code,start:start+len(ix)]=result.cpu().numpy()
                    if time.time()-last_progress>=30:
                        save(out/(mode+'_'+tags[0]+'_progress.json'),dict(pid=os.getpid(),mode=mode,current=tag,view=code,completed=start+len(ix),total=len(y),timestamp=time.time()));last_progress=time.time()
        if mode=='clean':np.savez_compressed(dest/'clean_scores.npz',ids=ids,labels=y,scores=scores[0])
        else:np.savez_compressed(dest/'view_scores.npz',ids=ids,labels=y,scores=scores,view_codes=np.arange(8))
        save(dest/(mode+'_receipt.json'),dict(source_commit=commit,source_files=hashes,profile_sha256=sha(profile),pid=os.getpid(),
            model_tag=tag,checkpoint_sha256=row['sha256'],metadata_sha256=sha(src/'metadata.json'),source_counter=counter,
            top_k=row['top_k'],seed=a['seed'],data_sha256=c['data_sha256'],views_forwarded=codes,
            images_forwarded=len(codes)*len(y),seconds=time.time()-started,teacher_train_mode=False,
            train_only=True,no_val_test_forward=True,no_optimizer=True,
            scores_sha256=sha(dest/('clean_scores.npz' if mode=='clean' else 'view_scores.npz'))))
        del model;torch.cuda.empty_cache()


def run(c,out,profile):
    import numpy as np
    assert list(c['models'])==['teacher17','teacher27','teacher37','parent']
    assert c['gpu_jobs']=={'1':['teacher17','teacher27'],'3':['teacher37','parent']}
    assert c['gpus']=={'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}
    assert (c['minimum_complement'],c['minimum_classes'],c['minimum_per_class'])==(64,4,4)
    commit,hashes=sources(profile,c);out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(configuration=c,source_commit=commit,source_files=hashes,pid=os.getpid(),host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    running={};children=[]
    def stop(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for mode in ['clean','full']:
            for index,tags in c['gpu_jobs'].items():
                gpu=c['gpus'][index];assert subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==gpu
                assert int(subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=7000
                log=(out/(mode+'_gpu'+index+'.log')).open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.ensemble_teacher_cache','--profile',str(profile),'--out',str(out),'--worker','--mode',mode,'--tags',','.join(tags)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[p.pid]=(p,log);children.append(p.pid)
            while running:
                for pid,(p,log) in list(running.items()):
                    if p.poll() is None:continue
                    log.close();del running[pid]
                    if p.returncode:raise RuntimeError('Cache producer failed '+str(pid))
                save(out/'status.json',dict(state='clean_train_audit' if mode=='clean' else 'caching_train_views',scheduler_pid=os.getpid(),running=list(running),timestamp=time.time()))
                if running:time.sleep(10)
            loaded={}
            for tag in c['models']:
                p=out/tag/('clean_scores.npz' if mode=='clean' else 'view_scores.npz')
                with np.load(p,allow_pickle=False) as z:
                    if not loaded:ids=z['ids'].copy();y=z['labels'].copy()
                    assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels']);loaded[tag]=z['scores'].copy()
            teacher=np.mean(np.stack([loaded[t] for t in ['teacher17','teacher27','teacher37']]),axis=0,dtype=np.float32)
            if mode=='clean':
                audit=admission(y,teacher,loaded['parent'],c['minimum_complement'],c['minimum_classes'],c['minimum_per_class'])
                audit.update(source_commit=commit,weights='Three teachers equal1/3, fixed before TRAIN audit',no_val_test_forward=True)
                save(out/'train_admission.json',audit)
                if not audit['admitted']:
                    save(out/'status.json',dict(state='rejected_by_train_audit',scheduler_pid=os.getpid(),running=[],reason='Insufficient TRAIN complement/distribution',audit=audit));return
            else:
                assert teacher.shape==(8,3496,8) and np.isfinite(teacher).all() and np.isfinite(loaded['parent']).all()
                assert np.max(abs(teacher.sum(-1)-1))<3e-6
                np.savez_compressed(out/'targets.npz',ids=ids,labels=y,teacher_scores=teacher,parent_scores=loaded['parent'],view_codes=np.arange(8))
                save(out/'cache_manifest.json',dict(source_commit=commit,source_files=hashes,configuration=c,
                    targets_sha256=sha(out/'targets.npz'),data_sha256=c['data_sha256'],train_samples=3496,view_codes=list(range(8)),
                    teacher_weights=[1/3]*3,average_across_teachers_not_views=True,train_only=True,no_val_test_forward=True,
                    model_score_sha256={k:sha(out/k/'view_scores.npz') for k in loaded},images_forwarded_total=4*8*3496))
                save(out/'status.json',dict(state='complete',scheduler_pid=os.getpid(),running=[],train_admitted=True,targets_sha256=sha(out/'targets.npz')))
    except BaseException as e:
        save(out/'status.json',dict(state='failed',scheduler_pid=os.getpid(),error=repr(e)));raise
    finally:
        for p,log in running.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        save(out/'release_check.json',dict(child_pids=children,children_reaped=True,authorized_gpus=c['gpus'],timestamp=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--mode',choices=['clean','full']);ap.add_argument('--tags');a=ap.parse_args();c=json.loads(a.profile.read_text())
    if a.worker:produce(c,a.out,a.profile,a.tags.split(','),a.mode)
    else:run(c,a.out,a.profile)
