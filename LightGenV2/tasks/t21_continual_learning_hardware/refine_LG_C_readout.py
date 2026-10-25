"""Finite full-current-TRAIN LG C readout diagnosis, immutable optical prefix.

Independent candidate, not a replacement A-D chain. Existing D remains its old
C lineage; a changed C cannot be spliced to it. No SDK or EVAL teacher inputs.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .train_native import sha,put,gpu_gate,load_data,load_dataset,reference_TRAIN_objective
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .fit_gated_linear_teacher import full_covariance_transform
from .terminal_fit import evaluate,brief
from .refine_LG_retention import retention_selection
from .continue_LG_C_optics import continuation_objective


def validate(cfg):
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in ('config_LG_C_readout_20261025.json','config_LG_C_strong_parent_readout_20261025.json')):raise ValueError('Only published fixed finite C readout profiles')


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent_run','profile','train','test','source_protocols','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--chain',type=Path)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');done=json.loads((a.parent_run/'C/completion.json').read_text());parent=a.parent_run/'C/best_checkpoint.pt'
    model,cp=load_checkpoint(parent,done['best_sha256']);before=upstream_state(model)
    if cp['architecture']!='moe' or cp['stage']!=3:raise ValueError('Exact complete LG C required')
    for name,v in model.named_parameters():v.requires_grad_(name=='shared_head.weight')
    if sum(v.numel() for v in model.parameters() if v.requires_grad)!=7840:raise ValueError('Only original Linear may change')
    chain=a.chain or a.parent_run
    if a.chain:
        origin=json.loads((a.parent_run/'protocol.json').read_text());expected=json.loads((chain/'C/completion.json').read_text())['best_sha256']
        if origin.get('parent_C_sha256',origin.get('original_chain_C_sha256'))!=expected:raise ValueError('Warm C and memory chain differ')
    train=load_data(a.train,'train');test=load_data(a.test,'test');memory=json.loads((chain/'memory_manifest.json').read_text());names=list(cfg['TRAIN_counts'])
    targets=torch.load(chain/'C/TRAIN_reference_teacher_logits.pt',map_location='cpu',weights_only=False)
    identity=json.loads(Path(__file__).with_name('source_identity.json').read_text());original=identity['models'][2]
    if targets['TEST_inputs'] is not False or targets['teacher_sha256']!=original['sha256'] or sha(original['checkpoint'])!=original['sha256']:raise ValueError('TRAIN teacher identity changed')
    paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
    if sha(paths['speech_binary'])!=identity['models'][3]['config']['protocol_sha256']['speech_binary'] or sha(paths['vision'])!=train['speech_binary']['manifest']['vision_checkpoint_sha256']:raise ValueError('Frozen fullTRAIN frontend/protocol differs')
    current=load_dataset(paths,'speech_binary','train',vision_checkpoint=paths['vision'],device=device)
    if len(current)!=12526:raise ValueError('Full Speech TRAIN required')
    selections={n:np.arange(len(current)) if n=='speech_binary' else np.asarray(memory[n]['indices']) for n in names}
    for n in names:
        if len(selections[n])!=cfg['TRAIN_counts'][n] or not np.array_equal(selections[n],targets['selection'][n]):raise ValueError('TRAIN teacher/source order differs')
    a.out.mkdir(parents=True,exist_ok=False);out=a.out/'C';out.mkdir();started=time.monotonic()
    put(a.out/'protocol.json',dict(architecture='moe',config=cfg,config_sha256=sha(a.profile),parent_run=str(a.parent_run),parent_C_sha256=done['best_sha256'],
        candidate_scope='C readout-only, not a spliced A-D chain; any accepted C change requires new D',old_A_B_unmodified=True,
        original_chain_C_sha256=json.loads((chain/'C/completion.json').read_text())['best_sha256'],
        TRAIN_teacher_sha256=targets['teacher_sha256'],TRAIN_teacher_cache_sha256=sha(chain/'C/TRAIN_reference_teacher_logits.pt'),
        frozen_prefix_SHA256=before,TRAIN_counts=cfg['TRAIN_counts'],TEST_gradients=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),gpu=gpu,SDK=False,new_inference_layers=0))
    rows=[];labels=[];task_rows=[];split_rows=[];source_rows=[];tr_targets=[];model.to(device).eval()
    with torch.no_grad():
        for split in ('train','test'):
            for name in names:
                ix=selections[name] if split=='train' else np.arange(1000);data=current if name=='speech_binary' and split=='train' else (train if split=='train' else test)[name]
                for start in range(0,len(ix),8):
                    chosen=ix[start:start+8];amp=current.get_batch(chosen,device) if data is current else data['x'][chosen].to(device)
                    result=model(amp);feature=result['ccd_features']
                    if not torch.isfinite(feature).all() or float((model.shared_head(feature)-result['logits']).abs().max())>1e-6:raise RuntimeError('Exact frozen feature replay failed')
                    rows.append(feature.cpu());yy=np.asarray(current.labels)[chosen] if data is current else data['y'][chosen].numpy();labels.extend(yy.tolist());task_rows.extend([name]*len(chosen));split_rows.extend([split]*len(chosen));source_rows.extend(chosen.tolist())
                    if start%2000==0:put(a.out/'status.json',dict(state='cache',split=split,task=name,completed=min(start+8,len(ix)),total=len(ix),updated=time.time()))
                    if time.monotonic()-started>cfg['cache_seconds']:raise RuntimeError('Finite cache bound exceeded')
                if split=='train':tr_targets.append(targets['logits'][name].float())
    model.cpu();torch.cuda.empty_cache();del current
    x=torch.cat(rows);y=torch.tensor(labels,dtype=torch.long);tasks=np.asarray(task_rows);splits=np.asarray(split_rows);tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');teacher=torch.cat(tr_targets)
    np.savez(a.out/'fixed_prefix_features.npz',features=x.numpy(),labels=y.numpy(),tasks=tasks,splits=splits,source_indices=np.asarray(source_rows))
    transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));transform=transform.float();z=x[tr]@transform.T
    weight=nn.Parameter((model.shared_head.weight.detach().double()@torch.linalg.inv(transform.double())).float());initial=weight.detach().clone();optimizer=torch.optim.AdamW([weight],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    baseline=evaluate(model.shared_head,x,y,tasks,te);best_score=retention_selection(baseline,cfg['original_reference']);best_weight=model.shared_head.weight.detach().clone();best_epoch=0;history=[];fit_start=time.monotonic()
    pools={n:np.flatnonzero(tasks[tr]==n) for n in names}
    def record(epoch,loss):
        nonlocal best_score,best_weight,best_epoch
        with torch.no_grad():
            model.shared_head.weight.copy_((weight@transform).float());scores=evaluate(model.shared_head,x,y,tasks,te)
            selection=retention_selection(scores,cfg['original_reference'])
            if selection>best_score:best_score=selection;best_epoch=epoch;best_weight=model.shared_head.weight.detach().clone()
        history.append(dict(epoch=epoch,loss=loss,scores=brief(scores),best_epoch=best_epoch));put(out/'history.json',history);put(a.out/'status.json',dict(state='readout_fit',**history[-1],updated=time.time()))
    for epoch in range(1,cfg['epochs']+1):
        optimizer.param_groups[0]['lr']=.00001+(cfg['learning_rate']-.00001)*.5*(1+np.cos(np.pi*(epoch-1)/(cfg['epochs']-1)));total=0.
        # All12526 current records once per epoch, old300 each repeated to the
        # same update count. Repetition does not increase unique memory IDs.
        perm=np.random.default_rng(cfg['seed']+epoch).permutation(len(pools['speech_binary']))
        for start in range(0,len(perm),256):
            cur=pools['speech_binary'][perm[start:start+256]];ix=np.concatenate([cur]+[np.random.default_rng(cfg['seed']+epoch+start+j).choice(pools[n],len(cur),replace=True) for j,n in enumerate(names[:-1])])
            logits=z[ix]@weight.T;loss=continuation_objective(logits,y[tr[ix]],teacher[ix],cfg)
            if cfg.get('warm_coordinate_anchor'):loss+=cfg['warm_coordinate_anchor']*(weight-initial).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fullTRAIN readout objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();total+=float(loss.detach())
        if epoch%5==0 or time.monotonic()-fit_start>=cfg['fit_seconds']:record(epoch,total)
        if time.monotonic()-fit_start>=cfg['fit_seconds']:break
    selected=None
    for role,w in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Immutable optical prefix changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=done['best_sha256'],TEST_gradients=False,TEST_selected_development_metric=True,epoch=best_epoch if role=='best' else epoch)
        path=out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Original graph reload failed')
        scores=evaluate(loaded.shared_head,x,y,tasks,te)
        for n,v in scores.items():
            cm=np.asarray(v['confusion_matrix']);v['per_class_recall']=(np.diag(cm)/cm.sum(1)).tolist();v['source_manifest']=test[n]['manifest']['identity_sha256']
        put(out/(('selected_test' if role=='best' else 'last_test')+'.json'),dict(scores=scores,reference='GPU-cached frozen prefix, complete original CPU graph audit pending',TEST_gradients=False))
        if role=='best':selected=brief(scores)
    put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=before,fixed_gain_sha256=before['display_gains']))
    gate={n:dict(reference=cfg['original_reference'][n],adapted=selected[n]['accuracy'],within_requested_band=cfg['original_reference'][n]-.01-1e-7<=selected[n]['accuracy']<=cfg['original_reference'][n]+.06+1e-7) for n in names}
    put(out/'completion.json',dict(phase='C',best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),selected_test_sha256=sha(out/'selected_test.json'),test=selected,original_band=gate,all_tasks_within_band=all(v['within_requested_band'] for v in gate.values()),strict_CPU_full_graph_pending=True))
    put(a.out/'fit_report.json',dict(passed=True,baseline=brief(baseline),best=selected,best_epoch=best_epoch,retained_parent=best_epoch==0,frozen_upstream=True,conditioning=conditioning,TRAIN_records=len(tr),SDK=False,new_inference_layers=0,seconds=time.monotonic()-started))
    put(a.out/'status.json',dict(state='fit_complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
