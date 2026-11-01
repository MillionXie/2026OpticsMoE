"""A finite same-architecture optical teacher followed by optical self-distillation.

Teacher hard supervision includes original TEST by explicit human instruction.
Student inherits those teacher weights; TEST labels never enter student loss.
All outcomes and checkpoint selection are development, not independent TEST.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .train_figure_revision import build_model, configure_parameters, training_forward, augment_packed, evaluate
from .figure_revision_data import load_panel_f
from .model_four import encode_rgb
from .phase_optimizer import WrappedPhaseAdam
from .alternating_readout import refit_existing_head
from .original_f_teacher import teacher_KL
from .f_load_balance import modality_mean_power_KL
from .report_f_load_balance import usage_statistics
from .gpu_policy import require_authorized_gpu
from .f_router_load_recovery import prepare_error_replay,replay_order
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def matched_balance(output, domains):
    """Hard-frequency stop-gradient weights plus differentiable marginal coverage."""
    hard = output['route_power']; dense = output['dense_route_power']
    if hard.shape != dense.shape or not bool(((hard > 0).sum(1) == 2).all()):
        raise ValueError('Exactly two actual selected experts required')
    terms = []
    for domain in (0, 1):
        mask = domains == domain
        if bool(mask.any()):
            frequency = (hard[mask] > 0).to(dense.dtype).mean(0).detach() / 2
            terms.append(4 * (frequency * dense[mask].mean(0)).sum())
    return torch.stack(terms).mean() + modality_mean_power_KL(dense, domains)


def classification_loss(output, labels, targets=None, soft_only=False):
    if soft_only:
        if labels is not None or targets is None:
            raise ValueError('Student original TEST requires only detached soft targets')
        return .5 * teacher_KL(output['raw_logits'], targets.detach(), 2.)
    if labels is None:
        raise ValueError('Hard-supervised teacher/TRAIN requires explicit labels')
    value = F.cross_entropy(output['raw_logits'], labels, label_smoothing=.02)
    if targets is not None:
        value = value + .5 * teacher_KL(output['raw_logits'], targets.detach(), 2.)
    return value


def selection_key(metrics, usage):
    # Accuracy first; among equal domain accuracies prefer real hard-load balance.
    accuracies = [metrics['domains'][name]['accuracy'] for name in ('RGB', 'SAR')]
    return min(accuracies), sum(accuracies), -max(v['selection_share_CV'] for v in usage.values()), -metrics['cross_entropy']


def matched_training_codes(rng, count, canonical_probability=.5):
    if not 0<=canonical_probability<=1:raise ValueError('Canonical probability outside [0,1]')
    codes=rng.integers(0,8,count);codes[rng.random(count)<canonical_probability]=0
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); cfg = json.loads(args.profile.read_text()); task = Path(__file__).resolve().parent
    if cfg['main_layers'] != 2 or cfg['architecture'] != 'LG' or cfg.get('top_k', 2) != 2 or cfg['dataset'] != 'eurosat_original_f':
        raise ValueError('Matched two-main-layer Top2 optical teacher and student required')
    if not cfg['teacher'].get('original_TEST_hard_GT_in_loss') or cfg['student'].get('original_TEST_hard_GT_in_loss'):
        raise ValueError('Explicit teacher TEST supervision and student soft-only disclosure required')
    if cfg['teacher']['epochs'] > 40 or cfg['student']['epochs'] > 20:
        raise ValueError('Finite teacher/student budgets required')
    head_codes=cfg.get('head_block_view_codes',list(range(8)))
    if not head_codes or len(set(head_codes))!=len(head_codes) or any(type(c)is not int or c not in range(8)for c in head_codes):raise ValueError('Explicit valid unique head view codes required')
    gpu = require_authorized_gpu(cfg['authorized_gpu_uuids']); torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed']); args.out.mkdir(parents=True, exist_ok=False)
    raw, identity = load_panel_f(cfg['data']);curriculum=prepare_error_replay(raw,cfg,task)
    fields = {s: torch.cat([encode_rgb(torch.from_numpy(v['images'][i:i+64]).cuda()) for i in range(0, len(v['labels']), 64)]) for s, v in raw.items()}
    labels = {s: torch.tensor(v['labels'], device='cuda', dtype=torch.long) for s, v in raw.items()}
    domains = {s: torch.tensor(v['domains'], device='cuda', dtype=torch.long) for s, v in raw.items()}
    for v in raw.values(): del v['images']
    parent = task / cfg['parent_checkpoint']
    if sha256(parent) != cfg['parent_checkpoint_sha256']: raise ValueError('Parent SHA')
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    model = build_model(cfg).cuda(); model.load_state_dict(payload['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 487204: raise ValueError('Matched model parameter count')
    target_cache = None
    try:
        for role in ('teacher', 'student'):
            out = args.out / role; out.mkdir(); phases, heads = configure_parameters(model, cfg)
            metadata = dict(profile=cfg, data=identity, role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT', stage=role,
                source_commit=os.environ.get('T17_SOURCE_COMMIT'), source_sha256={n: sha256(task/n) for n in
                [Path(__file__).name, 'train_figure_revision.py', 'model_four.py', 'model_linear_readout.py', 'model_depth_ccd.py', 'phase_optimizer.py', 'f_load_balance.py','f_router_load_recovery.py']},
                pid=os.getpid(), gpu_uuid=gpu, command=sys.argv, teacher_architecture_identical=True, inference_parameters=487204,
                TEST_hard_GT_in_current_stage_loss=role=='teacher', student_inherits_TEST_hard_supervised_teacher_weights=role=='student',
                VAL_gate=False, initial_checkpoint_sha256=sha256(parent))
            if curriculum:metadata['error_replay']=curriculum['record']
            if role == 'student': metadata['initial_checkpoint_sha256'] = sha256(args.out/'teacher/best_checkpoint.pt')
            write(out/'metadata.json', metadata); write(out/'status.json', dict(state='training', epoch=0, pid=os.getpid()))
            phase_opt=WrappedPhaseAdam(phases,cfg['phase_lr']); head_opt=torch.optim.AdamW(heads, lr=cfg['head_lr'], weight_decay=0)
            anchor=model.shared_head.weight.detach().clone(); budget=cfg[role]; history=[]; stale=0; selected=0
            fixed_router=model.router_phase.detach().clone() if cfg['training_scope']=='all_except_router' else None
            def metrics():
                dev, arrays=evaluate(model,fields['test'],labels['test'],cfg['eval_batch'],raw['test']['domains'])
                usage={name:usage_statistics(arrays['route_power'][arrays['domains']==d],arrays['dense_route_power'][arrays['domains']==d]) for d,name in [(0,'RGB'),(1,'SAR')]}
                return dev, usage
            def save(filename, epoch):
                torch.save(dict(model=model.state_dict(),epoch=epoch,metadata=metadata,optimizer=phase_opt.state_dict(),head_optimizer=head_opt.state_dict(),rng_state=rng.bit_generator.state),out/filename)
            dev, usage=metrics();metadata['initial_current_forward_development']=dev
            bootstrap=int(cfg.get('initial_teacher_head_bootstrap_steps',0))if role=='teacher'else 0
            if not 0<=bootstrap<=100:raise ValueError('Bounded teacher-only head bootstrap required')
            if bootstrap:
                features=[];truth=[];all_idx=[];offset=0;before={n:p.detach().clone()for n,p in model.named_parameters()if n!='shared_head.weight'};model.eval()
                with torch.no_grad():
                    for split in ('train','test'):
                        for i in range(0,len(fields[split]),cfg['eval_batch']):features.append(model.optical_features(fields[split][i:i+cfg['eval_batch']])['ccd_features'])
                        truth.append(labels[split]);indices=replay_order(np.random.default_rng(17),curriculum['wrong'][split])if curriculum else np.arange(len(fields[split]));all_idx.append(torch.tensor(indices+offset,device='cuda'));offset+=len(fields[split])
                idx=torch.cat(all_idx);record=refit_existing_head(model.shared_head,torch.cat(features)[idx],torch.cat(truth)[idx],anchor,.001,.02,bootstrap)
                if not all(torch.equal(p,before[n])for n,p in model.named_parameters()if n in before):raise RuntimeError('Bootstrap changed optical phase')
                metadata['teacher_head_bootstrap']=dict(actual=record,unique_original_TRAIN=6000,unique_original_TEST=2000,TEST_hard_GT=True,all_phase_bitwise_frozen=True)
                if cfg.get('bootstrap_reanchors_head'):anchor=model.shared_head.weight.detach().clone();metadata['teacher_head_bootstrap']['new_head_anchor_is_bootstrapped_weight']=True
                dev,usage=metrics()
            best=selection_key(dev,usage);save('best_checkpoint.pt',0)
            # Fixed coefficient from one balanced original TRAIN probe, no TEST/VAL calibration.
            ids=torch.tensor([int(np.flatnonzero((raw['train']['labels']==c)&(raw['train']['domains']==d))[0]) for d in (0,1) for c in range(10)],device='cuda')
            if fixed_router is not None:
                alpha=0.;metadata['balance_calibration']=dict(weight=0.,reason='Router frozen after explicit actual-load pretraining; usage cannot drift during classifier recovery')
            else:
                model.train(); output=training_forward(model,fields['train'][ids],cfg,True)
                base=classification_loss(output,labels['train'][ids]); balance=matched_balance(output,domains['train'][ids])
                g=torch.autograd.grad(base,model.router_phase,retain_graph=True)[0]; h=torch.autograd.grad(balance,model.router_phase)[0]
                jac=2*torch.pi*model.router_phase.sigmoid()*(1-model.router_phase.sigmoid())
                alpha=float((.5*(g/jac.clamp_min(1e-8)).norm()/(h/jac.clamp_min(1e-8)).norm().clamp_min(1e-12)).clamp(.001,1.))
                metadata['balance_calibration']=dict(weight=alpha,source='20 original TRAIN class-domain-balanced code0',target_physical_router_norm_ratio=.5)
            write(out/'metadata.json',metadata);save('best_checkpoint.pt',0)
            initial_experts=model.first_phase.detach().clone()
            for epoch in range(1,budget['epochs']+1):
                started=time.time(); model.train(); ratio=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,budget['epochs']-1)))
                for opt, lr in [(phase_opt,cfg['phase_lr']),(head_opt,cfg['head_lr'])]:
                    for group in opt.param_groups: group['lr']=lr*ratio
                parts={}; steps=0;expert_gradient_L2=np.zeros(4)
                for split in ('train','test'):
                    order=replay_order(rng,curriculum['wrong'][split]) if curriculum else rng.permutation(len(fields[split]));codes=matched_training_codes(rng,len(order),cfg.get('canonical_probability',.5))
                    total=0.; aux=0.
                    for start in range(0,len(order),cfg['batch']):
                        idx=torch.tensor(order[start:start+cfg['batch']],device='cuda'); code=torch.tensor(codes[start:start+cfg['batch']],device='cuda')
                        phase_opt.zero_grad(set_to_none=True);head_opt.zero_grad(set_to_none=True)
                        output=training_forward(model,augment_packed(fields[split][idx],code),cfg,True)
                        target=target_cache[split][idx,code] if role=='student' else None
                        hard=labels[split][idx] if role=='teacher' or split=='train' else None
                        main=classification_loss(output,hard,target,soft_only=role=='student' and split=='test')
                        regularizer=.0005*(model.shared_head.weight-anchor).square().sum()
                        balance=matched_balance(output,domains[split][idx]); loss=main+alpha*balance+regularizer; loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in phases+heads): raise RuntimeError('Invalid phase/head gradient')
                        expert_gradient_L2+=model.first_phase.grad.detach().flatten(1).norm(dim=1).cpu().numpy()
                        torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True);phase_opt.step();head_opt.step();steps+=1
                        total+=float(main.detach())*len(idx);aux+=float(balance.detach())*len(idx)
                    parts[split]=dict(images=len(order),unique_original_images=len(fields[split]),extra_error_replay_presentations=len(order)-len(fields[split]),updates=(len(order)+cfg['batch']-1)//cfg['batch'],hard_GT_in_loss=role=='teacher' or split=='train',mean_classification_loss=total/len(order),mean_balance_loss=aux/len(order),canonical_or_identity_view_presentations=int((codes==0).sum()),order_sha256=__import__('hashlib').sha256(order.astype('<i8').tobytes()).hexdigest(),D4_codes_sha256=__import__('hashlib').sha256(codes.astype('<i8').tobytes()).hexdigest())
                block=None
                if epoch%5==0:
                    features=[];targets=[];ys=[];masks=[];model.eval();before={n:p.detach().clone() for n,p in model.named_parameters() if n!='shared_head.weight'}
                    with torch.no_grad():
                        for split in ('train','test'):
                            for code in head_codes:
                                for i in range(0,len(fields[split]),cfg['eval_batch']):
                                    x=fields[split][i:i+cfg['eval_batch']];features.append(model.optical_features(augment_packed(x,torch.full((len(x),),code,device='cuda',dtype=torch.long)))['ccd_features'])
                                hard=role=='teacher' or split=='train';ys.append(labels[split] if hard else torch.full_like(labels[split],-1));masks.append(torch.full_like(labels[split],hard,dtype=torch.bool))
                                if role=='student':targets.append(target_cache[split][:,code])
                    X=torch.cat(features);Y=torch.cat(ys);M=torch.cat(masks);T=torch.cat(targets)if targets else None
                    if curriculum:
                        index=[];offset=0
                        for split in ('train','test'):
                            for code in head_codes:
                                index.append(torch.tensor(replay_order(np.random.default_rng(17),curriculum['wrong'][split])+offset,device='cuda'));offset+=len(fields[split])
                        index=torch.cat(index);X=X[index];Y=Y[index];M=M[index];T=T[index]if T is not None else None
                    block=refit_existing_head(model.shared_head,X,Y,anchor,.001,.02,40,T,.5 if targets else 0.,2.,M)
                    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before)
                    train_count=(6000+2*int(curriculum['wrong']['train'].sum())if curriculum else 6000)*len(head_codes);test_count=(2000+2*int(curriculum['wrong']['test'].sum())if curriculum else 2000)*len(head_codes)
                    block.update(total_view_presentations=train_count+test_count,unique_original_TRAIN_images=6000,unique_original_TEST_images=2000,hard_TRAIN_view_presentations=train_count,hard_TEST_view_presentations=test_count if role=='teacher' else 0,soft_TEST_view_presentations=test_count if role=='student' else 0,augmentation_codes=head_codes);head_opt.state.clear()
                if fixed_router is not None and not torch.equal(model.router_phase,fixed_router):raise RuntimeError('Frozen balanced router drifted')
                dev,usage=metrics();key=selection_key(dev,usage)
                if key>best:best=key;selected=epoch;stale=0;save('best_checkpoint.pt',epoch)
                else:stale+=1
                row=dict(epoch=epoch,selected_epoch=selected,original_test_development_selection=dev,development_expert_usage=usage,parts=parts,phase_head_updates=steps,balance_weight=alpha,seconds=time.time()-started)
                row['classification_raw_expert_gradient_L2_sum']=expert_gradient_L2.tolist();row['expert_phase_cumulative_raw_update_L2']=(model.first_phase.detach()-initial_experts).flatten(1).norm(dim=1).cpu().tolist();row['router_frozen_bitwise']=fixed_router is not None
                if block:row['head_block']=block
                history.append(row);write(out/'history.json',history);save('last_checkpoint.pt',epoch);write(out/'status.json',dict(state='training',epoch=epoch,selected_epoch=selected,pid=os.getpid()));print(json.dumps(dict(stage=role,**row)),flush=True)
                if epoch>=budget['min_epochs'] and stale>=budget['patience']:break
            model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True)
            result=dict(role=metadata['role'],stage=role,completed_epochs=epoch,selected_epoch=selected,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),teacher_architecture_identical=True,inference_parameters=487204)
            for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
                result[key],arrays=evaluate(model,fields[split],labels[split],cfg['eval_batch'],raw[split]['domains']);np.savez_compressed(out/f'{split}_predictions.npz',ids=raw[split]['ids'],**arrays)
            write(out/'result.json',result);write(out/'selection_lock.json',dict(selection='worst TEST domain then mean accuracy then actual load CV then CE; no VAL gate',checkpoint_sha256=result['checkpoint_sha256'],TEST_evaluations_before_lock=epoch+1));write(out/'status.json',dict(state='complete',pid=os.getpid(),**result))
            if role=='teacher':
                target_cache={};manifest=dict(teacher_checkpoint_sha256=result['checkpoint_sha256'],teacher_architecture_identical=True,teacher_TEST_hard_supervision=True,source_commit=metadata['source_commit'],splits={})
                model.eval()
                with torch.no_grad():
                    for split in ('train','test'):
                        views=[]
                        for code in range(8):
                            logits=[]
                            for i in range(0,len(fields[split]),cfg['eval_batch']):
                                x=fields[split][i:i+cfg['eval_batch']];logits.append(model(augment_packed(x,torch.full((len(x),),code,device='cuda',dtype=torch.long)))['raw_logits'])
                            views.append(torch.cat(logits))
                        target_cache[split]=torch.stack(views,1).detach();path=out/f'{split}_teacher_logits.npz';np.savez_compressed(path,ids=raw[split]['ids'],labels=raw[split]['labels'],domains=raw[split]['domains'],logits=target_cache[split].cpu().numpy());manifest['splits'][split]=dict(shape=list(target_cache[split].shape),sha256=sha256(path),view_codes=list(range(8)))
                write(out/'teacher_cache_manifest.json',manifest)
                parent=out/'best_checkpoint.pt'
        write(args.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:
        torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
