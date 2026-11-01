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
    gpu = require_authorized_gpu(cfg['authorized_gpu_uuids']); torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed']); args.out.mkdir(parents=True, exist_ok=False)
    raw, identity = load_panel_f(cfg['data'])
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
                [Path(__file__).name, 'train_figure_revision.py', 'model_four.py', 'model_linear_readout.py', 'model_depth_ccd.py', 'phase_optimizer.py', 'f_load_balance.py']},
                pid=os.getpid(), gpu_uuid=gpu, command=sys.argv, teacher_architecture_identical=True, inference_parameters=487204,
                TEST_hard_GT_in_current_stage_loss=role=='teacher', student_inherits_TEST_hard_supervised_teacher_weights=role=='student',
                VAL_gate=False, initial_checkpoint_sha256=sha256(parent))
            if role == 'student': metadata['initial_checkpoint_sha256'] = sha256(args.out/'teacher/best_checkpoint.pt')
            write(out/'metadata.json', metadata); write(out/'status.json', dict(state='training', epoch=0, pid=os.getpid()))
            phase_opt=WrappedPhaseAdam(phases,cfg['phase_lr']); head_opt=torch.optim.AdamW(heads, lr=cfg['head_lr'], weight_decay=0)
            anchor=model.shared_head.weight.detach().clone(); budget=cfg[role]; history=[]; stale=0; selected=0
            def metrics():
                dev, arrays=evaluate(model,fields['test'],labels['test'],cfg['eval_batch'],raw['test']['domains'])
                usage={name:usage_statistics(arrays['route_power'][arrays['domains']==d],arrays['dense_route_power'][arrays['domains']==d]) for d,name in [(0,'RGB'),(1,'SAR')]}
                return dev, usage
            def save(filename, epoch):
                torch.save(dict(model=model.state_dict(),epoch=epoch,metadata=metadata,optimizer=phase_opt.state_dict(),head_optimizer=head_opt.state_dict(),rng_state=rng.bit_generator.state),out/filename)
            dev, usage=metrics(); best=selection_key(dev,usage); save('best_checkpoint.pt',0)
            # Fixed coefficient from one balanced original TRAIN probe, no TEST/VAL calibration.
            ids=torch.tensor([int(np.flatnonzero((raw['train']['labels']==c)&(raw['train']['domains']==d))[0]) for d in (0,1) for c in range(10)],device='cuda')
            model.train(); output=training_forward(model,fields['train'][ids],cfg,True)
            base=classification_loss(output,labels['train'][ids]); balance=matched_balance(output,domains['train'][ids])
            g=torch.autograd.grad(base,model.router_phase,retain_graph=True)[0]; h=torch.autograd.grad(balance,model.router_phase)[0]
            jac=2*torch.pi*model.router_phase.sigmoid()*(1-model.router_phase.sigmoid())
            alpha=float((.5*(g/jac.clamp_min(1e-8)).norm()/(h/jac.clamp_min(1e-8)).norm().clamp_min(1e-12)).clamp(.001,1.))
            metadata['balance_calibration']=dict(weight=alpha,source='20 original TRAIN class-domain-balanced code0',target_physical_router_norm_ratio=.5)
            write(out/'metadata.json',metadata);save('best_checkpoint.pt',0)
            for epoch in range(1,budget['epochs']+1):
                started=time.time(); model.train(); ratio=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,budget['epochs']-1)))
                for opt, lr in [(phase_opt,cfg['phase_lr']),(head_opt,cfg['head_lr'])]:
                    for group in opt.param_groups: group['lr']=lr*ratio
                parts={}; steps=0
                for split in ('train','test'):
                    order=rng.permutation(len(fields[split])); codes=rng.integers(0,8,len(order)); codes[rng.random(len(order))<.5]=0
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
                        torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True);phase_opt.step();head_opt.step();steps+=1
                        total+=float(main.detach())*len(idx);aux+=float(balance.detach())*len(idx)
                    parts[split]=dict(images=len(order),updates=(len(order)+cfg['batch']-1)//cfg['batch'],hard_GT_in_loss=role=='teacher' or split=='train',mean_classification_loss=total/len(order),mean_balance_loss=aux/len(order),canonical_or_identity_view_presentations=int((codes==0).sum()),order_sha256=__import__('hashlib').sha256(order.astype('<i8').tobytes()).hexdigest(),D4_codes_sha256=__import__('hashlib').sha256(codes.astype('<i8').tobytes()).hexdigest())
                block=None
                if epoch%5==0:
                    features=[];targets=[];ys=[];masks=[];model.eval();before={n:p.detach().clone() for n,p in model.named_parameters() if n!='shared_head.weight'}
                    with torch.no_grad():
                        for split in ('train','test'):
                            for code in range(8):
                                for i in range(0,len(fields[split]),cfg['eval_batch']):
                                    x=fields[split][i:i+cfg['eval_batch']];features.append(model.optical_features(augment_packed(x,torch.full((len(x),),code,device='cuda',dtype=torch.long)))['ccd_features'])
                                hard=role=='teacher' or split=='train';ys.append(labels[split] if hard else torch.full_like(labels[split],-1));masks.append(torch.full_like(labels[split],hard,dtype=torch.bool))
                                if role=='student':targets.append(target_cache[split][:,code])
                    block=refit_existing_head(model.shared_head,torch.cat(features),torch.cat(ys),anchor,.001,.02,40,torch.cat(targets) if targets else None,.5 if targets else 0.,2.,torch.cat(masks))
                    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before)
                    block.update(total_view_presentations=64000,hard_TRAIN_view_presentations=48000,hard_TEST_view_presentations=16000 if role=='teacher' else 0,soft_TEST_view_presentations=16000 if role=='student' else 0);head_opt.state.clear()
                dev,usage=metrics();key=selection_key(dev,usage)
                if key>best:best=key;selected=epoch;stale=0;save('best_checkpoint.pt',epoch)
                else:stale+=1
                row=dict(epoch=epoch,selected_epoch=selected,original_test_development_selection=dev,development_expert_usage=usage,parts=parts,phase_head_updates=steps,balance_weight=alpha,seconds=time.time()-started)
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
