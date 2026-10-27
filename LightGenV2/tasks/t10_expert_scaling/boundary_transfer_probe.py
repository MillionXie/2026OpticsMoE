"""Nine source-reset TRAIN probes: opposing confusions and difficulty-matched transfer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import boundary_refine as base
from .phase_repair_probe import select_groups, rgb_views, phase_direction, sha_tensor
from .expert_tail_schedule import save,sha


def matched_groups(labels,scores):
    import numpy as np
    y=np.asarray(labels);pred=scores.argmax(1);logs=np.log(np.maximum(scores,1e-12));wrong=logs.copy();wrong[np.arange(len(y)),y]=-np.inf
    margin=logs[np.arange(len(y)),y]-wrong.max(1);old=set(select_groups(y,scores)['repair']);groups={};pairs={}
    for name,a,b in [('s',1,4),('d',4,1)]:
        ix=np.array([i for i in np.flatnonzero((y==a)&(pred==b)&(margin>=-.5)&(margin<0)) if i not in old],dtype=np.int64)
        ix=ix[np.argsort(-margin[ix],kind='stable')];n=min(8,len(ix)//2);assert n>0
        pair=ix[:n*2].reshape(n,2);coin=np.random.default_rng(17+a*100003).integers(0,2,size=n)
        groups['fit_'+name]=pair[np.arange(n),coin].tolist();groups['observe_'+name]=pair[np.arange(n),1-coin].tolist();pairs[name]=pair.tolist()
    for name,k in [('s',1),('d',4),('c',2)]:
        chosen=[]
        for j,(lo,hi) in enumerate([(0.,.2),(.2,.5)]):
            ix=np.flatnonzero((y==k)&(pred==k)&(margin>lo)&(margin<=hi));np.random.default_rng(17+k*100003+j).shuffle(ix)
            assert len(ix)>=8;chosen.extend(ix[:8].tolist())
        groups['guard_'+name]=chosen
    flat=sum(groups.values(),[]);assert len(flat)==len(set(flat)) and set(flat).isdisjoint(old)
    return groups,pairs


def transfer_metrics(scores,labels,tags,parent):
    import numpy as np
    p=scores.argmax(1);old=parent.argmax(1);logs=np.log(np.maximum(scores,1e-12));source=np.log(np.maximum(parent,1e-12))
    other=logs.copy();other[np.arange(len(labels)),labels]=-np.inf
    margin=logs[np.arange(len(labels)),labels]-other.max(1);result={}
    for group in dict.fromkeys(tags.tolist()):
        for view in [0,1]:
            ix=np.flatnonzero((tags==group)&(np.arange(len(tags))%2==view));y=labels[ix]
            competitors=np.where(y==1,4,1)
            if group=='guard_c':
                pair=logs[ix,2]-logs[ix][:,[1,4]].max(1);before=source[ix,2]-source[ix][:,[1,4]].max(1)
            else:pair=logs[ix,y]-logs[ix,competitors];before=source[ix,y]-source[ix,competitors]
            result[group+('_clean' if view==0 else '_d4')]=dict(samples=len(ix),correct=int((p[ix]==y).sum()),corrected=int(((old[ix]!=y)&(p[ix]==y)).sum()),
                new_wrong=int(((old[ix]==y)&(p[ix]!=y)).sum()),mean_margin=float(margin[ix].mean()),mean_pair_margin=float(pair.mean()),pair_margin_change=float((pair-before).mean()))
    return result


def run(c,out,profile):
    import numpy as np
    import torch
    from .train import data
    assert c['amplitudes']==[.0015,.005,.01] and c['seed']==17
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'
    commit=os.environ['T10_TRANSFER_SOURCE_COMMIT'];root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip());hashes={}
    paths=[Path(__file__),profile,Path(c['boundary_profile']),*[Path(__file__).with_name(n) for n in ['phase_repair_probe.py','boundary_refine.py','model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py']],Path(__file__).parent/'configs/study.json',root/'LightGenV2/common/optical_contract.py']
    for p in paths:
        n=p.resolve().relative_to(root).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+n]);hashes[n]=sha(p)
    out.mkdir(parents=True,exist_ok=False);start_time=time.time();torch.set_num_threads(4);torch.manual_seed(17)
    cfg=json.loads(Path(c['boundary_profile']).read_text());model,m,source_epoch=base.load_model(cfg);model.eval()
    assert sha(Path(m['arguments']['data']))==m['data_sha256'];splits,_=data(Path(m['arguments']['data']));train=splits['train']
    assert sha(Path(c['reference_train_scores']))==c['reference_train_scores_sha256']
    with np.load(c['reference_train_scores'],allow_pickle=False) as z:
        assert np.array_equal(z['ids'],train[2]) and np.array_equal(z['labels'],train[1].numpy());groups,pairs=matched_groups(z['labels'],z['scores']);reference_scores=z['scores'].copy()
    ix=np.array(sum(groups.values(),[]));rows=np.repeat(ix,2);tags=np.repeat(np.array(sum(([n]*len(v) for n,v in groups.items()),[])),2);y=train[1].numpy()[rows]
    with np.load(m['arguments']['data'],allow_pickle=False) as z:assert set(train[2][ix]).isdisjoint(z['test_ids'].astype(str))
    assert set(train[2][ix]).isdisjoint(splits['val'][2]);codes=np.zeros(len(rows),np.int64);codes[1::2]=np.random.default_rng(17).integers(1,8,size=len(ix))
    for n,p in model.named_parameters():p.requires_grad_(n=='global_phases.1')
    phase=model.global_phases[-1];anchor=phase.detach().clone();frozen={n:sha_tensor(p) for n,p in model.named_parameters() if n!='global_phases.1'}
    save(out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,configuration=c,source_epoch=source_epoch,source_checkpoint_sha256=cfg['reference_sha256'],data_sha256=m['data_sha256'],
        groups={n:train[2][v].tolist() for n,v in groups.items()},matched_pairs={n:train[2][v].tolist() for n,v in pairs.items()},selection='Source TRAIN margins only; old32 repair fit excluded; adjacent pairs split by fixed seed; correct guards stratified8+8. No VAL/TEST forward or optimizer trajectory.'))
    fields=np.empty((len(rows),model.c,model.c),np.float32);rp=[];parent=[]
    for start in range(0,len(rows),2):
        rgb=rgb_views(train,rows[start:start+2],codes[start:start+2])
        with torch.no_grad():
            f,r=base.prefix(model,rgb);p=base.suffix(model,f,r)['probabilities']
            if start==0:assert torch.equal(p,model(rgb)['probabilities'])
        fields[start:start+2]=f.cpu().numpy();rp.append(r.cpu().numpy());parent.append(p.cpu().numpy())
        save(out/'status.json',dict(state='train_fields',pid=os.getpid(),completed=min(start+2,len(rows)),total=len(rows),timestamp=time.time()))
    rp=np.concatenate(rp);parent=np.concatenate(parent);assert np.max(abs(parent[::2]-reference_scores[ix]))<1e-6
    np.savez_compressed(out/'train_identity.npz',ids=train[2][rows],indices=rows,labels=y,groups=tags,view_codes=codes,parent=parent)
    def forward(order):return base.suffix(model,torch.from_numpy(fields[order]).cuda(),torch.from_numpy(rp[order]).cuda())['probabilities']
    grads={}
    for group in ['fit_s','fit_d']:
        subset=np.flatnonzero(tags==group);g=torch.zeros_like(phase)
        for start in range(0,len(subset),2):
            order=subset[start:start+2];p=forward(order);logp=p.clamp_min(1e-12).log();labels=torch.from_numpy(y[order]).long().cuda();margin=logp.gather(1,labels[:,None]).squeeze(1)-logp.scatter(1,labels[:,None],float('-inf')).max(1).values
            loss=(.2-margin).clamp_min(0).square().mean();one,=torch.autograd.grad(loss,phase);g.add_(one,alpha=len(order)/len(subset))
        assert torch.isfinite(g).all() and float(g.norm())>0;grads[group]=g
        save(out/'status.json',dict(state='class_gradient',group=group,pid=os.getpid(),timestamp=time.time()))
    gs,gd=grads['fit_s'],grads['fit_d'];cos=float((gs*gd).sum()/(gs.norm()*gd.norm()));directions={'repair_s':phase_direction(gs),'repair_d':phase_direction(gd)}
    common=gs/gs.norm()+gd/gd.norm()
    if float(common.norm())>1e-6:directions['common']=phase_direction(common)
    directional={name:{k:float((d*g).sum()) for k,g in grads.items()} for name,d in directions.items()}
    baseline=transfer_metrics(parent,y,tags,parent);trials=[]
    for name,direction in directions.items():
        for amplitude in c['amplitudes']:
            with torch.no_grad():phase.copy_(anchor+amplitude*direction)
            rms=base.project_phase(phase,anchor,.02);scores=[]
            with torch.no_grad():
                for start in range(0,len(rows),2):scores.append(forward(np.arange(start,min(start+2,len(rows)))).cpu().numpy())
            scores=np.concatenate(scores);assert np.isfinite(scores).all() and abs(scores.sum(1)-1).max()<1e-5
            stats=transfer_metrics(scores,y,tags,parent);np.savez_compressed(out/(name+'_'+str(amplitude)+'_scores.npz'),scores=scores)
            trials.append(dict(direction=name,amplitude=amplitude,circular_rms=rms,groups=stats));save(out/'progress.json',dict(pid=os.getpid(),completed=len(trials),total=len(directions)*3,latest=trials[-1],timestamp=time.time()))
            with torch.no_grad():phase.copy_(anchor)
            assert torch.equal(phase,anchor)
    assert all(sha_tensor(p)==frozen[n] for n,p in model.named_parameters() if n in frozen)
    result=dict(source_commit=commit,gradient_cosine=cos,gradient_norms={n:float(g.norm()) for n,g in grads.items()},directional_derivatives=directional,baseline=baseline,trials=trials,
        source_restored_exactly=True,frozen_parameter_sha256=frozen,no_val_test_forward=True,unique_train_images=len(ix),train_views=len(rows),elapsed_seconds=time.time()-start_time)
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(json.loads(a.profile.read_text()),a.out,a.profile)
    except Exception as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),timestamp=time.time()))
        raise
