"""Two actual retained-Adam directions, source-restored block effects on TRAIN only."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from .expert_tail_schedule import save,sha
from .checkpoint_test_selection import model_hash
from .ensemble_teacher_cache import d4


def group_contains(name,block):
    return block=='full' or (block=='router' and name=='router_phase') or (block=='expert' and name.startswith('expert_phases.')) or (block=='global' and name.startswith('global_phases.'))


def admits_router(cases,noise):
    threshold=max(1e-5,10*noise)
    return len(cases)==2 and all(row['repair_margin_delta']>threshold and row['guard_new_wrong']==0 and row['guard_delta_q10']>=-threshold for case in cases for row in case)


def run(c,out,profile):
    import numpy as np
    import torch
    from . import reheat_refine as base
    from .model import ScalingOptics
    from .train import objective
    from .diagnose_topk import classification_margin_loss
    from .sharpness_refine import optimization_step
    from .cnn_distill import temperature_kl_part
    from .cached_distill import conditional_kl_part
    assert c['seed']==17 and c['fit_batches']==2 and c['fit_batch']==16 and c['heldout_per_class_per_group']==4
    assert c['blocks']==['router','expert','global','full'] and c['no_val_test_forward']
    assert os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid']
    assert subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['gpu_uuid']
    os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_BLOCK_SOURCE_COMMIT'];commit,hashes=base.sources(profile,c)
    source=Path(c['source_run']);assert sha(source/'last_checkpoint.pt')==c['source_last_sha256']
    last=torch.load(source/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert last['round']==16
    m=json.loads((source/'metadata.json').read_text());protocol=copy.deepcopy(m['protocol']);protocol['training_proposal']['phase_noise_std']=0.
    a=m['arguments'];assert a['experts']==a['top_k']==49 and a['layers']==4
    data=Path(a['data']);assert sha(data)==c['data_sha256']
    with np.load(data,allow_pickle=False) as z:x=z['train_images'].transpose(0,3,1,2).copy();y=z['train_labels'].astype(np.int64);ids=z['train_ids'].astype(str)
    target_root=Path(c['cnn_targets']);manifest=json.loads((target_root/'cache_manifest.json').read_text());assert manifest['source_commit']==c['teacher_cache_source_commit'];assert sha(target_root/'targets.npz')==manifest['targets_sha256']
    assert manifest['configuration']['models']['parent']['sha256']==c['objective_parent_sha256']
    with np.load(target_root/'targets.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],y)
        teacher=z['teacher_scores'];logits=z['teacher_logits'];parent=z['parent_scores']
    torch.set_num_threads(4);torch.manual_seed(17)
    model=ScalingOptics(protocol,49,49,'moe_oeo',8,a['layers'],a['padding']).cuda();model.load_state_dict(last['live_model'])
    assert sum(p.numel() for p in model.parameters())==11078432
    original={n:p.detach().cpu().clone() for n,p in model.state_dict().items()};original_hash=model_hash(original)
    assert original_hash==c['source_live_tensor_sha256']
    optimizer=torch.optim.AdamW([{'params':list(model.expert_phases),'lr_scale':1.,'name':'expert'},
        {'params':list(model.global_phases),'lr_scale':.5,'name':'global'},{'params':[model.router_phase],'lr_scale':.1,'name':'router'}],lr=3e-5,weight_decay=0.)
    opt_state=copy.deepcopy(last['optimizer']);optimizer.load_state_dict(copy.deepcopy(opt_state));del last
    out.mkdir(parents=True,exist_ok=False);save(out/'metadata.json',dict(pid=os.getpid(),source_commit=commit,source_files=hashes,configuration=c,
        actual_state_field='live_model',actual_optimizer_state='Retained last16 AdamW state, not EMA optimizer or fresh reset',original_model_tensor_sha256=original_hash,
        original_learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},cnn_targets_sha256=manifest['targets_sha256'],no_val_test_forward=True))
    def images(ix,codes):
        b=torch.from_numpy(x[ix]).float().cuda()/255
        for code in range(8):
            sel=np.flatnonzero(codes==code)
            if len(sel):
                j=torch.from_numpy(sel).to(b.device);b[j]=d4(b[j],int(code))
        return b
    def predict(ix,codes,tag=None):
        model.eval();rows=[]
        with torch.no_grad():
            for start in range(0,len(ix),4):
                rows.append(model(images(ix[start:start+4],codes[start:start+4]))['probabilities'].cpu().numpy())
                if tag and start%64==0:save(out/'status.json',dict(state=tag,pid=os.getpid(),completed=min(start+4,len(ix)),total=len(ix),timestamp=time.time()))
        return np.concatenate(rows)
    all_ix=np.arange(3496);reference=predict(all_ix,np.zeros(3496,dtype=np.int64),'TRAIN_reference');np.savez_compressed(out/'reference_train_scores.npz',ids=ids,labels=y,scores=reference)
    def margins(scores,labels):
        v=np.log(scores.clip(1e-12));true=v[np.arange(len(labels)),labels].copy();v[np.arange(len(labels)),labels]=-np.inf;return true-v.max(1)
    margin=margins(reference,y);p=reference.argmax(1)==y;t=teacher[0].argmax(1)==y;rng=np.random.default_rng(170419);repair=[];guard=[]
    for k in range(8):
        choices=np.flatnonzero((y==k)&t&~p);repair.extend(rng.permutation(choices)[:4].tolist())
        choices=np.flatnonzero((y==k)&p);guard.extend(choices[np.argsort(margin[choices],kind='stable')][:4].tolist())
    held=np.array(repair+guard,dtype=np.int64);nrepair=len(repair);assert nrepair>=16 and len(set(held))==len(held)
    fit=[];used=set(held.tolist())
    for batch in range(2):
        ix=[]
        for k in range(8):
            choices=[int(v) for v in rng.permutation(np.flatnonzero(y==k)) if int(v) not in used];ix.extend(choices[:2]);used.update(choices[:2])
        fit.append(np.array(ix,dtype=np.int64))
    codes=[np.zeros(len(held),dtype=np.int64),(held%7+1).astype(np.int64)];baseline=[predict(held,v) for v in codes]
    repeat=[predict(held,v) for v in codes];noise=max(float(abs(a-b).max()) for a,b in zip(baseline,repeat))
    save(out/'sampling.json',dict(reference_train_ids=ids.tolist(),fit_indices=[v.tolist() for v in fit],fit_ids=[ids[v].tolist() for v in fit],
        repair_indices=repair,guard_indices=guard,heldout_ids=ids[held].tolist(),heldout_codes=[v.tolist() for v in codes],source_correct_clean=int(p.sum()),
        source_teacher_only_clean=int((t&~p).sum()),selection='Only source TRAIN scores/labels and fixed RNG; heldout excluded from both class-balanced fit batches'))
    weights=torch.ones(8,device='cuda');rows=[];router_cases=[]
    for no,ix in enumerate(fit,1):
        model.load_state_dict(original);optimizer.load_state_dict(copy.deepcopy(opt_state));model.train();fcodes=(ix+no)%8;eligible=teacher[fcodes,ix].argmax(1)==y[ix];protect=(parent[fcodes,ix].argmax(1)==y[ix])&~eligible
        def closure():
            total=0.
            for start in range(0,16,2):
                u=ix[start:start+2];vc=fcodes[start:start+2];labels=torch.from_numpy(y[u]).cuda();output=model(images(u,vc));prob=output['probabilities']
                loss=(objective(model,output,labels,weights,protocol['training_proposal'])+.2*classification_margin_loss(prob,labels))/8
                loss=loss+c['lambda_kd']*temperature_kl_part(prob,torch.from_numpy(logits[vc,u]).cuda(),torch.from_numpy(eligible[start:start+2]).cuda(),int(eligible.sum()))
                loss=loss+.5*conditional_kl_part(prob,torch.from_numpy(parent[vc,u]).cuda(),torch.from_numpy(protect[start:start+2]).cuda(),int(protect.sum()))
                loss.backward();total+=float(loss.detach())
            return total
        info=optimization_step(model,optimizer,closure);assert len(info['gradients'])==5 and all(v['finite'] and v['norm']>0 for v in info['gradients'].values())
        candidate={n:p.detach().cpu().clone() for n,p in model.state_dict().items()};deltas={n:dict(rms=float((candidate[n]-v).square().mean().sqrt()),max_abs=float((candidate[n]-v).abs().max())) for n,v in original.items()}
        case=dict(batch=no,step=info,parameter_delta=deltas,blocks={})
        for block in c['blocks']:
            model.load_state_dict({n:(candidate[n] if group_contains(n,block) else v) for n,v in original.items()});result=[]
            for view,vc in enumerate(codes):
                scores=predict(held,vc);before=margins(baseline[view],y[held]);after=margins(scores,y[held]);delta=after-before;correct=scores.argmax(1)==y[held];old_correct=baseline[view].argmax(1)==y[held]
                row=dict(view=view,repair_margin_delta=float(delta[:nrepair].mean()),repair_improved_fraction=float((delta[:nrepair]>1e-5).mean()),
                    guard_source_correct=int(old_correct[nrepair:].sum()),guard_current_correct=int(correct[nrepair:].sum()),
                    guard_new_wrong=int((old_correct[nrepair:]&~correct[nrepair:]).sum()),guard_delta_q10=float(np.quantile(delta[nrepair:],.1)),guard_delta_mean=float(delta[nrepair:].mean()))
                result.append(row);np.savez_compressed(out/f'batch{no}_{block}_view{view}.npz',ids=ids[held],labels=y[held],scores=scores,source_scores=baseline[view],view_codes=vc)
            case['blocks'][block]=result
        router_cases.append(case['blocks']['router']);rows.append(case);save(out/'status.json',dict(state='probing',pid=os.getpid(),completed=no,total=2,timestamp=time.time()))
    model.load_state_dict(original);optimizer.load_state_dict(copy.deepcopy(opt_state));assert model_hash(model.state_dict())==original_hash
    assert sha(source/'last_checkpoint.pt')==c['source_last_sha256']
    decision=admits_router(router_cases,noise)
    save(out/'receipt.json',dict(source_commit=commit,pid=os.getpid(),source_last_sha256=c['source_last_sha256'],source_live_tensor_sha256=original_hash,
        source_restored_exactly=True,optimizer_state_reloaded=True,source_disk_checkpoint_unchanged=True,fit_batches=2,actual_candidate_Adam_steps=2,
        source_restored_block_trials=8,reference_train_samples=3496,heldout_repair=nrepair,heldout_guard=len(guard),heldout_views=2,
        repeated_baseline_probability_max_error=noise,admission_threshold=max(1e-5,10*noise),cases=rows,router_positive_in_both_batches_and_views=decision,
        training_auto_started=False,no_val_test_forward=True,limitation='Only these actual end-of-stage Adam directions at their retained learning rates, not a proof of router capability or guaranteed transfer. No amplitude or lambda grid.'))
    save(out/'status.json',dict(state='complete',pid=os.getpid(),router_direction_admitted_for_consideration=decision,training_auto_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=json.loads(a.profile.read_text())
    try:run(c,a.out,a.profile)
    except BaseException as error:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error)))
        raise
