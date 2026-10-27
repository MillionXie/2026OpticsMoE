"""Frozen-prefix, cached last-optical-phase boundary refinement; TEST is development data."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save,sha


def validate(c):
    if (c['seed'],c['recipes'],c['segments'],c['views_per_segment'],c['batch'],c['microbatch']) != (17,['control','boundary'],12,512,16,2):
        raise ValueError('Fixed paired seed17 budget required')
    if c['development_checks'] != [4,8,12] or c['target_correct'] != 616:
        raise ValueError('Development selection contract changed')
    if (c['margin'],c['kl_easy'],c['kl_hard'],c['anchor_weight'],c['phase_rms_radius'],c['ema_decay']) != (.2,.5,.05,50.,.02,.95):
        raise ValueError('Declared boundary protection changed')
    if c['gpus'] != {'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}:
        raise ValueError('Only GPUs1/3')


def sources(profile):
    commit=os.environ['T10_BOUNDARY_SOURCE_COMMIT']
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    paths=[Path(__file__),profile,*[Path(__file__).with_name(n) for n in ['model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py']],Path(__file__).parent/'configs/study.json',root/'LightGenV2/common/optical_contract.py']
    hashes={}
    for p in paths:
        name=p.resolve().relative_to(root).as_posix()
        if p.read_bytes()!=subprocess.check_output(['git','show',commit+':'+name]):raise ValueError('Source changed: '+name)
        hashes[name]=sha(p)
    return commit,hashes


def pools(labels,scores,seed=17):
    import numpy as np
    labels=np.asarray(labels);logs=np.log(np.maximum(scores,1e-12));other=logs.copy();other[np.arange(len(labels)),labels]=-np.inf
    margin=logs[np.arange(len(labels)),labels]-other.max(1);groups=[]
    for group in range(3):
        chosen=[];hardflags=[]
        for k in range(8):
            ix=np.flatnonzero(labels==k);rank=ix[np.argsort(margin[ix],kind='stable')]
            hard=rank[group*8:group*8+8]
            remaining=np.setdiff1d(ix,hard);np.random.default_rng(seed+k*100003).shuffle(remaining)
            replay=np.take(remaining,np.arange(group*24,(group+1)*24)%len(remaining))
            chosen.extend(hard.tolist()+replay.tolist());hardflags.extend([True]*8+[False]*24)
        chosen=np.asarray(chosen,dtype=np.int64);hardflags=np.asarray(hardflags)
        assert len(set(chosen))==256 and np.bincount(labels[chosen],minlength=8).tolist()==[32]*8
        groups.append((chosen,hardflags))
    return groups


def lr(segment):
    if segment==1:return 5e-5
    return 2e-5+(1e-4-2e-5)*(1+math.cos(math.pi*(segment-2)/10))/2


def margin_and_protection(probabilities,labels,parent,c):
    import torch
    logp=probabilities.clamp_min(1e-12).log();logq=parent.clamp_min(1e-12).log()
    correct=logp.gather(1,labels[:,None]).squeeze(1)
    wrong=logp.scatter(1,labels[:,None],float('-inf')).max(1).values
    violation=(c['margin']+wrong-correct).clamp_min(0)
    qmargin=logq.gather(1,labels[:,None]).squeeze(1)-logq.scatter(1,labels[:,None],float('-inf')).max(1).values
    easy=(parent.argmax(1)==labels)&(qmargin>=c['margin'])
    weights=torch.where(easy,c['kl_easy'],c['kl_hard'])
    kl=(weights*(parent*(logq-logp)).sum(1)).mean()
    return violation,kl,easy


def project_phase(phase,anchor,radius):
    import torch
    with torch.no_grad():
        delta=torch.atan2((phase-anchor).sin(),(phase-anchor).cos());rms=delta.square().mean().sqrt()
        if float(rms)>radius:phase.copy_(anchor+delta*(radius/float(rms)))
        return float(torch.atan2((phase-anchor).sin(),(phase-anchor).cos()).square().mean().sqrt())


def prefix(model,rgb):
    import torch
    from .model import encode_rgb
    source=encode_rgb(rgb);amp,rp,mask=model.route(encode_rgb(rgb,model.router_side));assert torch.all(mask==1)
    field=model.assemble(source[:,None]*amp[:,:,None,None]).to(torch.complex64)
    phases=[(p,e) for pair in zip(model.expert_phases,model.global_phases) for p,e in zip(pair,(True,False))]
    for p,e in phases[:-1]:field=model.phase_step(field,p,e)
    assert float(field.imag.abs().max())==0.
    return field.real,rp


def suffix(model,field,rp):
    import torch
    field=model.phase_step(field.to(torch.complex64),model.global_phases[-1],False)
    intensity=model.prop(field).abs().square()
    energy=torch.stack([intensity[:,a:b,c:d].sum((-2,-1)) for a,b,c,d in model.detectors],1)
    return dict(probabilities=(energy+1e-12)/(energy.sum(1,keepdim=True)+model.classes*1e-12),
        capture=energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12),route_probabilities=rp,route_mask=torch.ones_like(rp))


def load_model(c):
    import torch
    from .model import ScalingOptics
    reference=Path(c['reference_run']);assert sha(reference/'best_checkpoint.pt')==c['reference_sha256']
    m=json.loads((reference/'metadata.json').read_text());a=m['arguments'];assert a['seed']==17 and a['experts']==a['top_k']==49 and a['layers']==4
    protocol=copy.deepcopy(m['protocol']);protocol['training_proposal']['phase_noise_std']=0.
    model=ScalingOptics(protocol,49,49,'moe_oeo',8,a['layers'],a['padding']).cuda()
    checkpoint=torch.load(reference/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'])
    return model,m,checkpoint['epoch']


def build_cache(c,out,profile):
    import numpy as np
    import torch
    from .train import data,images,evaluate
    assert os.environ['CUDA_VISIBLE_DEVICES']==c['gpus']['3']
    commit,hashes=sources(profile);torch.set_num_threads(4);model,m,_=load_model(c);model.eval();cache=Path(m['arguments']['data']);assert sha(cache)==m['data_sha256'];splits,manifest=data(cache)
    with np.load(cache,allow_pickle=False) as z:
        splits['test']=(torch.from_numpy(z['test_images'].transpose(0,3,1,2).copy()),torch.from_numpy(z['test_labels'].astype(np.int64)),z['test_ids'].astype(str))
    for name in ['train','val','test']:assert splits[name][2].tolist()==manifest['split_ids'][name]
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    save(out/'launch_metadata.json',dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,reference_sha256=c['reference_sha256'],data_sha256=m['data_sha256'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    save(out/'status.json',dict(state='reference_TRAIN_scoring',pid=os.getpid(),timestamp=time.time()))
    train,scores=evaluate(model,splits['train'],4)
    np.savez_compressed(out/'reference_train_scores.npz',ids=splits['train'][2],labels=splits['train'][1].numpy(),scores=scores.numpy())
    grouped=pools(splits['train'][1].numpy(),scores.numpy(),17);cases={};frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n!='global_phases.1'}
    old_test=json.loads((Path(c['reference_run'])/'test_result.json').read_text())
    assert sha(Path(c['reference_run'])/'test_predictions.npz')==old_test['predictions_sha256'] and old_test['checkpoint_sha256']==c['reference_sha256'] and old_test['data_sha256']==m['data_sha256']
    with np.load(Path(c['reference_run'])/'test_predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],splits['test'][2]);old_test_scores=z['scores'].copy()
    def one(name,split,indices,codes,hard=None):
        x,y,ids=split;n=len(indices);fieldpath=out/(name+'_fields.npy')
        array=np.lib.format.open_memmap(fieldpath,mode='w+',dtype=np.float32,shape=(n,model.c,model.c))
        q=np.empty((n,8),np.float32);route=np.empty((n,49),np.float32);last_progress=0.;max_error=0.
        with torch.no_grad():
            for start in range(0,n,2):
                ix=torch.from_numpy(indices[start:start+2]);rgb=images(x,ix,0,17)
                for j,code in enumerate(codes[start:start+2]):
                    if code>=4:rgb[j]=rgb[j].flip(-1)
                    rgb[j]=rgb[j].rot90(int(code)%4,(-2,-1))
                field,rp=prefix(model,rgb);array[start:start+len(ix)]=field.cpu().numpy();route[start:start+len(ix)]=rp.cpu().numpy()
                if name=='test':q[start:start+len(ix)]=old_test_scores[start:start+len(ix)]
                else:q[start:start+len(ix)]=suffix(model,field,rp)['probabilities'].cpu().numpy()
                if start==0:
                    a=suffix(model,field,rp)['probabilities'];b=model(rgb)['probabilities'];error=float((a-b).abs().max());assert error==0.;max_error=max(max_error,error)
                if time.time()-last_progress>30:
                    save(out/'status.json',dict(state='building',case=name,completed=start+len(ix),total=n,pid=os.getpid(),timestamp=time.time()));last_progress=time.time()
        array.flush();del array
        infopath=out/(name+'_info.npz');np.savez_compressed(infopath,ids=ids[indices],labels=y[indices].numpy(),indices=indices,view_codes=codes,parent_probabilities=q,route_probabilities=route,hard=np.zeros(n,bool) if hard is None else hard)
        cases[name]=dict(fields_file=fieldpath.name,fields_sha256=sha(fieldpath),info_file=infopath.name,info_sha256=sha(infopath),samples=n,unique_ids=len(set(ids[indices])),max_full_suffix_error=max_error)
    for group,(ix,hard) in enumerate(grouped):
        rows=np.repeat(ix,2);codes=np.zeros(512,dtype=np.int64);codes[1::2]=np.random.default_rng(17+group).integers(1,8,size=256)
        one('train'+str(group),splits['train'],rows,codes,np.repeat(hard,2))
    for name in ['val','test']:one(name,splits[name],np.arange(752,dtype=np.int64),np.zeros(752,dtype=np.int64))
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p.detach().cpu(),frozen[n])
    info=dict(source_commit=commit,source_files=hashes,profile_sha256=sha(profile),source_checkpoint_sha256=c['reference_sha256'],data_sha256=m['data_sha256'],canvas_side=model.c,
        cases=cases,train_reference=train,elapsed_seconds=time.time()-started,prefix_frozen_names=list(frozen),cache_contract='float32 real OEO amplitude before final global phase; exact fixed prefix, ids and explicit D4 codes; no TEST gradients')
    save(out/'cache_manifest.json',info);save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time(),elapsed_seconds=info['elapsed_seconds']));print(json.dumps(info),flush=True)


def cached_case(cache,manifest,name):
    import numpy as np
    c=manifest['cases'][name];fields=cache/c['fields_file'];info=cache/c['info_file']
    assert sha(fields)==c['fields_sha256'] and sha(info)==c['info_sha256']
    with np.load(info,allow_pickle=False) as z:small={k:z[k] for k in z.files}
    return np.load(fields,mmap_mode='r'),small


def metrics(scores,labels):
    import numpy as np
    pred=scores.argmax(1);cm=np.zeros((8,8),int);np.add.at(cm,(labels,pred),1)
    nll=-np.log(np.maximum(scores[np.arange(len(labels)),labels],1e-12))
    return dict(accuracy=float((pred==labels).mean()),macro_nll=float(np.mean([nll[labels==k].mean() for k in np.unique(labels)])),confusion_matrix=cm.tolist())


def evaluate_cached(model,case,indices=None):
    import numpy as np
    import torch
    fields,info=case;order=np.arange(len(info['labels'])) if indices is None else np.asarray(indices);rows=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(order),2):
            ix=order[start:start+2];field=torch.from_numpy(np.array(fields[ix],copy=True)).cuda();rp=torch.from_numpy(info['route_probabilities'][ix]).cuda();p=suffix(model,field,rp)['probabilities'];assert torch.isfinite(p).all();rows.append(p.cpu().numpy())
    scores=np.concatenate(rows);return metrics(scores,info['labels'][order]),scores


def worker(c,out,cache,profile,recipe):
    import numpy as np
    import torch
    from .train import data,evaluate
    assert recipe in c['recipes'] and os.environ['CUDA_VISIBLE_DEVICES']==c['gpus']['1' if recipe=='control' else '3']
    commit,hashes=sources(profile);torch.set_num_threads(4);torch.manual_seed(17);model,m,source_epoch=load_model(c)
    manifest=json.loads((cache/'cache_manifest.json').read_text());assert manifest['source_checkpoint_sha256']==c['reference_sha256'] and manifest['profile_sha256']==sha(profile) and manifest['data_sha256']==m['data_sha256']
    for n,p in model.named_parameters():p.requires_grad_(n=='global_phases.1')
    phase=model.global_phases[-1];assert phase.numel()==3055504 and sum(p.numel() for p in model.parameters() if p.requires_grad)==3055504
    anchor=phase.detach().clone();ema=anchor.clone();optimizer=torch.optim.AdamW([phase],lr=lr(1),weight_decay=0.)
    frozen_hashes={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters() if n!='global_phases.1'}
    out.mkdir(parents=True,exist_ok=False);reference=Path(c['reference_run']);shutil.copy2(reference/'best_checkpoint.pt',out/'best_checkpoint.pt')
    metadata=dict(pid=os.getpid(),command=sys.argv,git_commit=commit,source_files=hashes,protocol=model.cfg,arguments=dict(m['arguments'],out=str(out),epochs=12),data_sha256=m['data_sha256'],source_selected_epoch=source_epoch,
        reference_run=str(reference),reference_sha256=c['reference_sha256'],configuration=c,cache_manifest_sha256=sha(cache/'cache_manifest.json'),trainable_parameters=3055504,full_parameters=sum(p.numel() for p in model.parameters()),counter_unit='512 TRAIN-view segment, not full epoch',initialization='Selected EMA, fresh AdamW; no exact resume',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    save(out/'metadata.json',metadata);valcase=cached_case(cache,manifest,'val');testcase=cached_case(cache,manifest,'test')
    old=json.loads((reference/'test_result.json').read_text());best_correct=round(old['test']['accuracy']*752);best_nll=old['test']['macro_nll'];best_segment=0;history=[];started=time.time();last_progress=0.;current_group=-1
    baseline,_=evaluate_cached(model,valcase);assert abs(baseline['accuracy']-json.loads((reference/'result.json').read_text())['val']['accuracy'])<1.001/752
    # Cache identity check only: no new baseline TEST metric or baseline checkpoint selection.
    _,check=evaluate_cached(model,testcase,np.arange(4));assert np.max(np.abs(check-testcase[1]['parent_probabilities'][:4]))<1e-6
    def payload(segment,live):
        return dict(model=model.state_dict(),live_last_phase=live,optimizer=optimizer.state_dict(),epoch=segment,segment=segment,power_fraction=1.,metadata=metadata,
            rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state()))
    for segment in range(1,13):
        group=(segment-1)//4
        if group!=current_group:
            traincase=cached_case(cache,manifest,'train'+str(group));current_group=group
            assert len(set(traincase[1]['ids']))==256 and np.bincount(traincase[1]['labels'],minlength=8).tolist()==[64]*8
            assert set(traincase[1]['ids']).isdisjoint(valcase[1]['ids']) and set(traincase[1]['ids']).isdisjoint(testcase[1]['ids'])
        fields,info=traincase;order=np.random.default_rng(17+segment*100003).permutation(512);model.train();optimizer.param_groups[0]['lr']=lr(segment);totals={k:0. for k in ['supervised','kl','anchor','physics','total']};t0=time.time();easy_sum=0.;active_sum=0.
        with(out/'sampling_history.jsonl').open('a') as f:f.write(json.dumps(dict(segment=segment,group=group,cache_rows=order.tolist(),ids=info['ids'][order].tolist(),view_codes=info['view_codes'][order].tolist()))+'\n')
        for start in range(0,512,16):
            optimizer.zero_grad(set_to_none=True);batch=order[start:start+16]
            for offset in range(0,16,2):
                ix=batch[offset:offset+2];field=torch.from_numpy(np.array(fields[ix],copy=True)).cuda();rp=torch.from_numpy(info['route_probabilities'][ix]).cuda();labels=torch.from_numpy(info['labels'][ix].astype(np.int64)).cuda();q=torch.from_numpy(info['parent_probabilities'][ix]).cuda();output=suffix(model,field,rp)
                violation,kl,easy=margin_and_protection(output['probabilities'],labels,q,c);tc=model.cfg['training_proposal']
                physics=tc['capture_loss_weight']*(-output['capture'].clamp_min(1e-12).log()).mean()+tc['phase_tv_weight']*model.regularizer()+tc['router_balance_weight']*(49*rp.mean(0).square().sum()-1)
                if recipe=='control':
                    logp=output['probabilities'].clamp_min(1e-12).log();smooth=tc['label_smoothing']
                    supervised=-((1-smooth)*logp.gather(1,labels[:,None]).squeeze(1)+smooth*logp.mean(1)).mean()+.2*violation.mean()
                else:supervised=violation.square().mean()
                phase_anchor=c['anchor_weight']*(2*((phase-anchor)/2).sin().square()).mean();loss=supervised+kl+phase_anchor+physics
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite boundary objective')
                (loss/8).backward();parts=dict(supervised=supervised,kl=kl,anchor=phase_anchor,physics=physics,total=loss)
                for k,v in parts.items():totals[k]+=float(v.detach())/8
                easy_sum+=float(easy.float().mean())/8;active_sum+=float((violation>0).float().mean())/8
            norm=torch.nn.utils.clip_grad_norm_([phase],1.,error_if_nonfinite=True);assert float(norm)>0
            assert all(p.grad is None for n,p in model.named_parameters() if n!='global_phases.1')
            optimizer.step();rms=project_phase(phase,anchor,c['phase_rms_radius'])
            with torch.no_grad():ema.lerp_(phase,1-c['ema_decay'])
            if time.time()-last_progress>30:
                save(out/'batch_progress.json',dict(pid=os.getpid(),segment=segment,optimizer_batches=start//16+1,batches_per_segment=32,loss=totals['total']/(start//16+1),gradient_norm=float(norm),phase_rms=rms,timestamp=time.time()));last_progress=time.time()
        training_seconds=time.time()-t0;live=phase.detach().clone();live_val,_=evaluate_cached(model,valcase)
        with torch.no_grad():phase.copy_(ema)
        try:
            val,val_scores=evaluate_cached(model,valcase);train_pool,_=evaluate_cached(model,traincase,np.arange(0,512,2))
            torch.save(payload(segment,live),out/'last_checkpoint.pt');decision=None;dev=None
            if segment in c['development_checks']:
                save(out/f'development_lock_{segment:02}.json',dict(checkpoint_sha256=sha(out/'last_checkpoint.pt'),segment=segment,power_fraction=1.,selection='EMA candidate fixed before authorized development TEST'))
                dev,dev_scores=evaluate_cached(model,testcase);correct=round(dev['accuracy']*752)
                np.savez_compressed(out/f'development_test_{segment:02}_predictions.npz',ids=testcase[1]['ids'],labels=testcase[1]['labels'],scores=dev_scores,predictions=dev_scores.argmax(1))
                accepted=correct>best_correct or (correct==best_correct and dev['macro_nll']<best_nll-1e-4)
                decision=dict(correct=correct,nll=dev['macro_nll'],accepted=accepted,previous_correct=best_correct,previous_nll=best_nll)
                if accepted:
                    best_correct,best_nll,best_segment=correct,dev['macro_nll'],segment;shutil.copy2(out/'last_checkpoint.pt',out/'best_checkpoint.pt')
                    np.savez_compressed(out/'best_development_test_predictions.npz',ids=testcase[1]['ids'],labels=testcase[1]['labels'],scores=dev_scores,predictions=dev_scores.argmax(1))
                save(out/f'development_decision_{segment:02}.json',decision)
            entry=dict(segment=segment,group=group,optimizer_updates=segment*32,optimized_train_views=segment*512,train_clean_pool=train_pool,train_unique_pool=256,train_augmented_views=512,
                live_val=live_val,ema_val=val,development_test=dev,development_decision=decision,losses={k:v/32 for k,v in totals.items()},parent_easy_fraction=easy_sum/32,
                margin_active_fraction=active_sum/32,gradient_norm=float(norm),phase_rms_live=rms,learning_rate=lr(segment),training_seconds=training_seconds,elapsed_seconds=time.time()-started,
                best_segment=best_segment,best_development_correct=best_correct,prefix_frozen=True)
            history.append(entry);save(out/'history.json',history);save(out/'status.json',dict(state='training',pid=os.getpid(),segment=segment,best_segment=best_segment,best_development_correct=best_correct));print(json.dumps(entry),flush=True)
        finally:
            with torch.no_grad():phase.copy_(live)
    for n,p in model.named_parameters():
        if n in frozen_hashes:assert hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()==frozen_hashes[n]
    save(out/'prefix_identity_receipt.json',dict(frozen_parameter_sha256=frozen_hashes,unchanged_after_all_segments=True,cache_source_sha256=c['reference_sha256']))
    selected=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);del selected
    assert sha(Path(m['arguments']['data']))==m['data_sha256']
    splits,_=data(Path(m['arguments']['data']));train,_=evaluate(model,splits['train'],4);val,_=evaluate(model,splits['val'],4)
    if best_segment:
        with np.load(Path(m['arguments']['data']),allow_pickle=False) as z:testsplit=(torch.from_numpy(z['test_images'].transpose(0,3,1,2).copy()),torch.from_numpy(z['test_labels'].astype(np.int64)),z['test_ids'].astype(str))
        final_test,p=evaluate(model,testsplit,4)
        with np.load(out/'best_development_test_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['ids'],testsplit[2]);error=float(np.abs(z['scores']-p.numpy()).max());mismatches=int((z['scores'].argmax(1)!=p.argmax(1).numpy()).sum());assert error<1e-5
        np.savez_compressed(out/'test_predictions.npz',ids=testsplit[2],labels=testsplit[1].numpy(),scores=p.numpy(),predictions=p.argmax(1).numpy())
        save(out/'full_model_cache_equivalence.json',dict(max_probability_error=error,argmax_mismatches=mismatches,full_original_model_forward=True))
    else:
        final_test=old['test'];shutil.copy2(reference/'test_predictions.npz',out/'test_predictions.npz');assert sha(out/'best_checkpoint.pt')==c['reference_sha256']
        save(out/'test_reuse_receipt.json',dict(identical_checkpoint=True,gpu_forward_executed=False,reference=str(reference)))
    route_hashes={}
    for split in ['train','val']:
        name=f'selected_{split}_routes.npz'
        with np.load(reference/name,allow_pickle=False) as z:
            assert np.array_equal(z['ids'],splits[split][2]) and np.array_equal(z['labels'],splits[split][1].numpy())
            assert np.all(z['route_mask']==1) and np.isfinite(z['route_probabilities']).all()
        shutil.copy2(reference/name,out/name);route_hashes[name]=sha(out/name)
    save(out/'route_reuse_receipt.json',dict(reason='Router and prefix frozen at identical source; routes depend only on input/router, not final phase.',reference=str(reference),prefix_frozen=True,sha256=route_hashes))
    result=dict(recipe=recipe,seed=17,segments=12,selected_segment=best_segment,checkpoint_sha256=sha(out/'best_checkpoint.pt'),source_commit=commit,train=train,val=val,test=final_test,
        test_predictions_sha256=sha(out/'test_predictions.npz'),test_correct=round(final_test['accuracy']*752),target_correct=616,pilot_target_met=round(final_test['accuracy']*752)>=616,
        trainable_parameters=3055504,full_parameters=11078432,optimized_train_views=6144,optimizer_updates=384,role=c['role'],data_sha256=m['data_sha256'])
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),**result))


def supervisor(c,out,profile):
    validate(c);commit,hashes=sources(profile);out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(configuration=c,source_commit=commit,source_files=hashes,host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scheduler_pid=os.getpid()))
    children=[];running={};done=[];env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def stop(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(mode,directory,gpu,recipe=None):
        uuid=c['gpus'][str(gpu)];assert subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==uuid
        if int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<7000:raise RuntimeError('Insufficient authorized GPU memory')
        cmd=[sys.executable,'-u','-m',__package__+'.boundary_refine','--profile',str(profile),'--out',str(directory),'--mode',mode,'--cache',str(out/'cache')]
        if recipe:cmd+=['--recipe',recipe]
        log=directory.with_suffix('.log').open('x');p=subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(p.pid);running[p.pid]=(p,log,directory);return p
    try:
        p=launch('cache',out/'cache',3)
        while p.poll() is None:
            save(out/'status.json',dict(state='building_cache',scheduler_pid=os.getpid(),running={str(k):str(v[2]) for k,v in running.items()}));time.sleep(10)
        _,log,_=running.pop(p.pid);log.close()
        if p.returncode:raise RuntimeError('Cache producer failed')
        for recipe,gpu in [('control',1),('boundary',3)]:launch('worker',out/(recipe+'_s17'),gpu,recipe)
        while running:
            for pid,(p,log,directory) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[pid]
                if p.returncode:raise RuntimeError('Boundary worker failed '+str(directory))
                done.append(json.loads((directory/'result.json').read_text()))
            save(out/'status.json',dict(state='training',scheduler_pid=os.getpid(),running={str(k):str(v[2]) for k,v in running.items()},completed=done))
            if running:time.sleep(10)
        selected=max(done,key=lambda r:(r['test_correct'],-r['test']['macro_nll']));save(out/'summary.json',dict(candidates=done,selected=selected,pilot_target_met=selected['pilot_target_met'],target_correct=616,three_seed_mean_target_established=False,role=c['role']))
        save(out/'status.json',dict(state='complete',scheduler_pid=os.getpid(),running={},pilot_target_met=selected['pilot_target_met']))
    finally:
        for p,log,directory in running.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        save(out/'release_check.json',dict(child_pids=children,children_reaped=True,timestamp=time.time(),authorized_gpus=c['gpus']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cache',type=Path);ap.add_argument('--mode',choices=['supervisor','cache','worker'],default='supervisor');ap.add_argument('--recipe',choices=['control','boundary']);a=ap.parse_args();c=json.loads(a.profile.read_text());validate(c)
    try:
        if a.mode=='supervisor':supervisor(c,a.out,a.profile)
        elif a.mode=='cache':build_cache(c,a.out,a.profile)
        else:worker(c,a.out,a.cache,a.profile,a.recipe)
    except BaseException as error:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),scheduler_pid=os.getpid(),error=repr(error)))
        raise
