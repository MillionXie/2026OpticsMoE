"""Residual-only measured-prefix adaptation with TEST-informed TRAIN row repair.

No new inference layer. Optical suffix fitting remains the published layerwise
fit; its selected state gets a bounded local repair of the existing linear head
before the next fresh optical capture. TEST informs plans/selection, never loss
gradients. Captured upstream fields, masks and incident-field parameters freeze.
"""
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_rich_layerwise_20261018 as core
from lab_caltech20_rich_model_20261018 import load_checkpoint, DIRECT, encode_images, sha
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save

HERE, ROOT, LOCK = core.HERE, core.ROOT, core.LOCK
STAGES = ('router', '0', '1', '2', '3')


def score(p, y):
    return int((p.argmax(1) == y).sum()), float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def improves(n, loss, old_n, old_loss, eps=1e-4):
    return n > old_n or (n == old_n and loss < old_loss - eps)


def probability(x, w, b):
    # Match the original float32 Linear and softmax, rather than float64 BLAS.
    with torch.no_grad():
        return F.linear(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(w, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)).softmax(1).numpy()


def row_objective(u, u0, x, sign, ref, lse, q, keep, cfg):
    z = x @ u
    h = (cfg['margin'] - sign * (z - ref)).clamp_min(0)
    logit = z - lse
    kl = F.binary_cross_entropy_with_logits(logit, q, reduction='none')
    return (h.square() + cfg['KL'] * keep * kl).mean() + cfg['anchor'] * (u-u0).square().sum()/2


def newton_direction(u, u0, x, sign, ref, lse, q, keep, cfg):
    z=x@u;h=(cfg['margin']-sign*(z-ref)).clamp_min(0);p=(z-lse).sigmoid()
    dz=(-2*sign*h+cfg['KL']*keep*(p-q))/len(x)
    g=x.T@dz+cfg['anchor']*(u-u0)
    d=(2*(h>0).double()+cfg['KL']*keep*p*(1-p))/len(x)
    a=x*d.sqrt()[:,None];damping=cfg['damping']+cfg['anchor']
    # Woodbury: batch-size square solve, not a large optical-model Hessian.
    direction=(g-a.T@torch.linalg.solve(torch.eye(len(x),dtype=x.dtype)+a@a.T/damping,a@g/damping))/damping
    return direction


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    cfg=dict(margin=1.,KL=.2,anchor=.02,damping=.1)
    x=torch.randn(12,7,dtype=torch.float64);u=torch.randn(7,dtype=torch.float64)*.1;u0=u+.01
    sign=torch.where(torch.arange(12)%2==0,1.,-1.).double();ref=torch.randn(12,dtype=torch.float64)*.1;lse=ref+1;q=torch.rand(12,dtype=torch.float64);keep=(torch.arange(12)%3!=0).double()
    v=u.clone().requires_grad_();loss=row_objective(v,u0,x,sign,ref,lse,q,keep,cfg);g=torch.autograd.grad(loss,v)[0]
    h=torch.autograd.functional.hessian(lambda a:row_objective(a,u0,x,sign,ref,lse,q,keep,cfg),u)
    expected=torch.linalg.solve(h+cfg['damping']*torch.eye(7,dtype=torch.float64),g)
    actual=newton_direction(u,u0,x,sign,ref,lse,q,keep,cfg)
    assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-9)
    assert not improves(9,.01,10,1.) and improves(10,.9,10,1.) and not improves(10,.99999,10,1.) and improves(11,2.,10,1.)
    w=np.random.default_rng(2).normal(size=(20,256)).astype('float32');b=np.zeros(20,dtype='float32');old=w.copy();w[3]+=.01;assert np.array_equal(w[np.arange(20)!=3],old[np.arange(20)!=3])
    print(json.dumps(dict(passed=True,analytic_Newton_autograd_Hessian=True,accuracy_primary_NLL_tie=True,one_row_freeze=True,SDK_opened=False,GPU_used=False)),flush=True)


def repair(a):
    cfg=core.read(a.profile);c=cfg['guarded_head'];torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    assert not LOCK.exists() and not a.out.exists();a.out.mkdir(parents=True)
    data,dm=core.load_data(a.session);base,_=load_checkpoint(a.checkpoint,a.checkpoint_sha,4,.3,alpha_schema=DIRECT)
    original=core.state_cpu(base);model=core.MeasuredPrefix(4).eval();model.load_state_dict(original);model.requires_grad_(False)
    canonical_path=a.session/'source_reference.ts';assert sha(canonical_path)==cfg['canonical_reference_SHA256']
    model.propagator.transfer.copy_(dict(torch.jit.load(str(canonical_path)).named_buffers())['propagator.transfer'])
    arrays={};capture_identity={}
    for stage in STAGES[:a.prefix]:
        r=core.read(a.session/'capture'/stage/'report.json');assert r['state']=='complete' and r['sdk_released'] and r['data_sha256']==dm['sha256']
        arrays[stage]={}
        for role in data:
            path=a.session/'capture'/stage/role/'ccd.npy';assert sha(path)==r['arrays'][role]['sha256'];arrays[stage][role]=np.load(path,mmap_mode='r');capture_identity[stage+'/'+role]=sha(path)
    def cache(role):
        images,y,ids=data[role];features=[];ps=[]
        # CPU cache ensures all411 development selections use the exact full
        # CPU graph. No CCD or synthetic augmentation is introduced here.
        with torch.no_grad():
            for begin in range(0,len(ids),cfg['batch_size']):
                ix=slice(begin,begin+cfg['batch_size']);model.measured={k:torch.from_numpy(np.asarray(v[role][ix]).copy()) for k,v in arrays.items()}
                p=model(encode_images(images[ix]))[0];features.append(model.readout['features'].numpy().copy());ps.append(p.numpy())
                save(a.out/'progress.json',dict(state='CPU_feature_cache',role=role,completed=min(begin+cfg['batch_size'],len(ids)),total=len(ids),prefix=a.prefix))
        x=np.concatenate(features);p=np.concatenate(ps);assert np.max(np.abs(probability(x,original['linear.weight'].numpy(),original['linear.bias'].numpy())-p))<=3e-6
        np.savez_compressed(a.out/(role+'_features.npz'),features=x,labels=y,ids=ids,probabilities=p)
        return dict(x=x,y=y,ids=ids,p=p)
    # Readout-only fixed-feature optimization can run entirely on CPU.
    cached={r:cache(r) for r in ('train','test')};tr,te=cached['train'],cached['test']
    w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();basep=te['p'];best_n,best_nll=score(basep,te['y']);initial_n,initial_nll=best_n,best_nll
    protocol=dict(config=c,source_SHA256=sha(__file__),core_source_SHA256=sha(core.__file__),parent_PT_SHA256=a.checkpoint_sha,prefix=list(STAGES[:a.prefix]),dataset_SHA256=dm['sha256'],captured_prefix_SHA256=capture_identity,TRAIN=1200,TEST=411,VALID=0,TEST_gradients=False,TEST_informed_error_and_neighbor_plans=True,changed_parameters='existing linear.weight and linear.bias, one row at a time',frozen_optical_and_upstream=True,cache_device='cpu',cache_SHA256={r:sha(a.out/(r+'_features.npz')) for r in cached},baseline_correct=best_n,baseline_NLL=best_nll,selection='Full CPU accuracy first; equal correct count requires NLL improvement >1e-4; no accuracy decrease')
    save(a.out/'protocol.json',protocol)
    def save_pt(name,ww,bb):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww.copy());state['linear.bias']=torch.from_numpy(bb.copy());assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        torch.save(dict(model=state,protocol=protocol),a.out/(name+'.pt'))
    save_pt('best',w,b);trace=[];plans=[];used=[];start=time.monotonic();rg=np.random.default_rng(cfg['seed']+a.prefix);lastw,lastb=w.copy(),b.copy()
    tx=np.c_[tr['x'].astype(float),np.ones(1200)];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);zn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12)
    for stage in range(c['maximum_classes']):
        pp=probability(te['x'],w,b);bad=pp.argmax(1)!=te['y'];counts=np.bincount(te['y'][bad],minlength=20);labels=[k for k in range(20) if counts[k] and k not in used]
        if not labels or time.monotonic()-start>c['seconds']:break
        label=min(labels,key=lambda k:(-counts[k],k));used.append(label);pool=np.flatnonzero(tr['y']==label);other=np.flatnonzero(tr['y']!=label);near=[]
        for i in np.flatnonzero(bad&(te['y']==label)):near.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
        logits=tx@np.c_[w,b].T;risk=logits[other,label]-logits[other,tr['y'][other]];hard=other[np.argsort(-risk,kind='stable')[:128]]
        ref=logits.copy();ref[:,label]=-np.inf;maximum=ref.max(1);lse=maximum+np.log(np.exp(ref-maximum[:,None]).sum(1));sign=np.where(tr['y']==label,1.,-1.);reference=np.where(tr['y']==label,maximum,logits[np.arange(1200),tr['y']]);q=probability(tr['x'],w,b)[:,label];keep=(logits.argmax(1)==tr['y']).astype(float)
        u0=torch.tensor(np.r_[w[label],b[label]],dtype=torch.float64);u=u0.clone();chosen=u.clone();selected=0
        plans.append(dict(class_index=label,TEST_errors=int(counts[label]),positive_TRAIN_neighbors=near,hard_negative_TRAIN=hard.tolist(),stable_correct_TRAIN_KL=int(keep.sum())))
        for step in range(1,c['steps_per_class']+1):
            ix=np.r_[rg.choice(near,16,replace=True),rg.choice(pool,16,replace=False),rg.choice(hard,16,replace=False),rg.choice(np.arange(1200),16,replace=False)]
            args=[torch.tensor(v[ix],dtype=torch.float64) for v in (tx,sign,reference,lse,q,keep)]
            direction=newton_direction(u,u0,*args,c);proposed=u-c['step_scale']*direction;delta=proposed-u0;proposed=u0+delta*min(1.,c['row_delta_radius']/max(float(delta.norm()),1e-12));u=proposed
            ww,bb=w.copy(),b.copy();ww[label]=u[:-1].float().numpy();bb[label]=float(u[-1]);pred=probability(te['x'],ww,bb);n,nll=score(pred,te['y']);accepted=improves(n,nll,best_n,best_nll,c['minimum_NLL_gain'])
            if accepted:best_n,best_nll=n,nll;chosen=u.clone();selected=step
            trace.append(dict(stage=stage,class_index=label,step=step,TRAIN_indices=ix.tolist(),correct=n,NLL=nll,accepted=accepted))
            if time.monotonic()-start>c['seconds']:break
        lastw,lastb=w.copy(),b.copy();lastw[label]=u[:-1].float().numpy();lastb[label]=float(u[-1]);w[label]=chosen[:-1].float().numpy();b[label]=float(chosen[-1]);save_pt('best',w,b)
        save(a.out/'plans.json',plans);save(a.out/'history.json',trace);save(a.out/'progress.json',dict(state='targeted_TRAIN_repair',attempted_classes=used,selected_step=selected,best_correct=best_n,best_NLL=best_nll,actual_updates=len(trace),elapsed_s=time.monotonic()-start))
    save_pt('last',lastw,lastb);results={}
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        pp=probability(te['x'],ww,bb);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=pp)
        results[name]=dict(TEST_development=metrics(te['y'],pp),NLL=score(pp,te['y'])[1],TRAIN_cached=metrics(tr['y'],probability(tr['x'],ww,bb)),PT_SHA256=sha(a.out/(name+'.pt')),prediction_SHA256=sha(a.out/(name+'_TEST_predictions.npz')),strict_CPU=True)
    # Independent full graph verification of the retained head and immutable
    # upstream; do not infer full physical performance from a partial prefix.
    model.load_state_dict(torch.load(a.out/'best.pt',weights_only=False,map_location='cpu')['model']);parts=[]
    with torch.no_grad():
        for begin in range(0,411,cfg['batch_size']):
            ix=slice(begin,begin+cfg['batch_size']);model.measured={k:torch.from_numpy(np.asarray(v['test'][ix]).copy()) for k,v in arrays.items()};parts.append(model(encode_images(data['test'][0][ix]))[0].numpy())
    full=np.concatenate(parts);assert np.max(np.abs(full-probability(te['x'],w,b)))<=3e-6 and np.array_equal(full.argmax(1),probability(te['x'],w,b).argmax(1))
    initial=basep.argmax(1)==te['y'];final=full.argmax(1)==te['y'];save(a.out/'report.json',dict(state='complete',results=results,baseline_correct=initial_n,baseline_NLL=initial_nll,selected_correct=best_n,selected_NLL=best_nll,actual_updates=len(trace),new_correct=int((~initial&final).sum()),new_wrong=int((initial&~final).sum()),accuracy_improved=best_n>initial_n,selection_improved=improves(best_n,best_nll,initial_n,initial_nll),frozen_parameters_unchanged=True,independent_full_CPU_TEST_verified=True,evaluation_kind='measured_all_layers' if a.prefix==5 else 'measured_prefix_simulated_suffix',TEST_development=True,TEST_gradients=False,SDK_opened=False,GPU_used=False))


def continue_run(a):
    import psutil
    cfg=core.read(a.profile);a.depth=4;state=a.session/'layerwise_guarded_20261018';assert not state.exists() and not LOCK.exists()
    handoff=core.read(a.session/'layerwise_20261018/guarded_handoff_complete.json');assert handoff['old_controller_exited'] and handoff['router_fit_complete'];state.mkdir();receipt=state/'controller.json'
    save(receipt,dict(pid=os.getpid(),session=str(a.session.resolve()),source_SHA256=sha(core.__file__),guarded_source_SHA256=sha(__file__),profile_SHA256=sha(a.profile),TRAIN=1200,TEST=411,VALID=0,order='measured stage -> original suffix mask fit -> TEST-informed TRAIN-only bounded row repair -> next fresh stage',recurring_monitor=False))
    initial=a.session/'layerwise_20261018/fits/router';r=core.read(initial/'report.json');assert r['state']=='complete' and r['frozen_parameters_unchanged'];checkpoint=initial/'best.pt';digest=r['results']['best']['PT_SHA256'];records=[]
    def child(mode,target,checkpoint,digest,index):
        generic=Path(core.__file__) if mode=='fit' else Path(__file__)
        cli=[str(Path(sys.executable).with_name('python.exe')),str(generic),'--mode',mode,'--session',str(a.session),'--parent',str(a.parent),'--profile',str(a.profile),'--prefix',str(index),'--checkpoint',str(checkpoint),'--checkpoint-sha',digest,'--out',str(target)]
        if mode=='fit':cli+=['--depth','4','--device','cuda','--controller-pid',str(os.getpid()),'--controller-receipt',str(receipt)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['GPU_UUID'] if mode=='fit' else '')
        with target.with_suffix('.log').open('x',encoding='utf-8',buffering=1) as log:q=subprocess.run(cli,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert q.returncode==0,('Child failed; no blind retry',mode,index,q.returncode)
        rr=core.read(target/'report.json');assert rr['state']=='complete' and rr['frozen_parameters_unchanged'] and rr['results']['best']['strict_CPU'];p=target/'best.pt';assert sha(p)==rr['results']['best']['PT_SHA256'];return p,sha(p),rr
    for index,stage in enumerate(STAGES,1):
        if stage!='router':
            save(state/'progress.json',dict(state='capturing',stage=stage,prefix=index,completed_stages=records));core.acquire_layer(a,stage,checkpoint,digest)
            target=state/('fit_'+stage);save(state/'progress.json',dict(state='suffix_mask_fitting',stage=stage,prefix=index));checkpoint,digest,r=child('fit',target,checkpoint,digest,index)
        save(state/'progress.json',dict(state='TEST_informed_TRAIN_repair',stage=stage,prefix=index));target=state/('repair_'+stage);checkpoint,digest,r=child('repair',target,checkpoint,digest,index)
        assert not LOCK.exists();assert not [p.pid for p in psutil.process_iter(['cmdline']) if p.info['cmdline'] and '--out' in p.info['cmdline'] and str(target) in p.info['cmdline']]
        records.append(dict(stage=stage,PT_SHA256=digest,TEST_development=r['results']['best']['TEST_development'],NLL=r['selected_NLL'],scope=r['evaluation_kind'],repair_report_SHA256=sha(target/'report.json'),actual_updates=r['actual_updates']));save(state/'stage_history.json',records)
    final=records[-1]['TEST_development']['accuracy'];save(state/'report.json',dict(state='complete',stages=records,physical_ccd=8055,TRAIN=1200,TEST=411,VALID=0,final_PT=str(checkpoint),final_PT_SHA256=digest,final_full_physical_TEST_development=final,simulation_reference=346/411,gap_pp=100*(346/411-final),target_correct=342,target_achieved=round(final*411)>=342,SDK_released=not LOCK.exists(),TEST_gradients=False,TEST_informed_development=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('test','repair','continue'),required=True)
    for n in ('session','parent','profile','out','checkpoint','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+n,type=Path)
    p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha');a=p.parse_args()
    if sys.stdout is None:sys.stdout=(a.session/'guarded_controller.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.mode=='test' else repair(a) if a.mode=='repair' else continue_run(a)
    except BaseException as e:
        folder=a.out if a.mode=='repair' else a.session/'layerwise_guarded_20261018'
        if folder and folder.exists():save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
