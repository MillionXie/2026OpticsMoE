"""Finite joint F balance refinement using stable optical/training primitives.

Separate controller avoids changing the source loaded by the active panel-D
queue. No model copy, extra inference module, domain input or routing oracle.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from .train_figure_revision import build_model,configure_parameters,training_forward,augment_packed,evaluate,checkpoint_selection_key
from .f_joint_distillation import validate_joint_parent,prepare_joint_teacher,prepare_joint_test_targets
from .alternating_readout import anchored_classification,refit_existing_head
from .original_f_teacher import teacher_KL
from .f_load_balance import modality_mean_power_KL
from .figure_revision_data import load_panel_f
from .model_four import encode_rgb
from .phase_optimizer import WrappedPhaseAdam
from .gpu_policy import require_authorized_gpu
from .report_f_load_balance import usage_statistics
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def balanced_objective(output, labels, domains, targets, head, anchor, cfg, soft_only=False):
    """Soft-only TEST path rejects hard labels; acquisition domains only group loss."""
    logits=output['raw_logits'];teacher=cfg['teacher'];strength=cfg['alternating_head']['anchor_strength']
    if soft_only:
        if labels is not None:raise ValueError('Original TEST hard labels forbidden')
        base=teacher['weight']*teacher_KL(logits,targets,teacher['temperature'])+.5*strength*(head.weight-anchor).square().sum()
    else:
        if labels is None:raise ValueError('TRAIN labels required')
        base=anchored_classification(logits,labels,head.weight,anchor,strength,cfg['label_smoothing'],targets,teacher['weight'],teacher['temperature'])
    coefficient=cfg['per_modality_balance']['weight']
    if coefficient==0:return base,dict(main=float(base.detach()),balance=0.)
    balance=modality_mean_power_KL(output['dense_route_power'],domains)
    return base+coefficient*balance,dict(main=float(base.detach()),balance=float(balance.detach()))


def refit_joint_head(model, fields, targets, labels, anchor, cfg):
    features=[];teacher=[]
    model.eval()
    with torch.no_grad():
        for split in ('train','test'):
            for view in range(8):
                for start in range(0,len(fields[split]),cfg['eval_batch']):
                    x=fields[split][start:start+cfg['eval_batch']]
                    if view:x=augment_packed(x,torch.full((len(x),),view,device=x.device,dtype=torch.long))
                    features.append(model.optical_features(x)['ccd_features'])
                teacher.append(targets[split][:,view])
    hard=labels['train'].repeat(8);soft=torch.full((len(labels['test'])*8,),-1,device='cuda',dtype=torch.long)
    mask=torch.cat([torch.ones_like(hard,dtype=torch.bool),torch.zeros_like(soft,dtype=torch.bool)])
    before={k:v.detach().clone() for k,v in model.state_dict().items() if k!='shared_head.weight'}
    record=refit_existing_head(model.shared_head,torch.cat(features),torch.cat([hard,soft]),anchor,cfg['alternating_head']['anchor_strength'],cfg['label_smoothing'],cfg['alternating_head']['inner_steps'],torch.cat(teacher),cfg['teacher']['weight'],cfg['teacher']['temperature'],mask)
    if not all(torch.equal(model.state_dict()[k],v) for k,v in before.items()):raise RuntimeError('Head block changed phase')
    record.update(original_TRAIN_images=len(labels['train']),source_original_TEST_images=len(labels['test']),hard_TRAIN_view_presentations=len(hard),source_TEST_soft_view_presentations=len(soft),total_view_presentations=len(hard)+len(soft),augmentation_codes=list(range(8)))
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();cfg=json.loads(args.profile.read_text());task=Path(__file__).resolve().parent
    if cfg['selection_source']!='original_TEST_development' or cfg['training_scope']!='all' or cfg['epochs']>20 or cfg.get('training_domain') or cfg.get('domain_route_aux',0):raise ValueError('Explicit finite joint balance profile required')
    if cfg['per_modality_balance']['mode']!='modality_mean_power_KL' or not 0<cfg['per_modality_balance']['weight']<=1:raise ValueError('Invalid explicit balance loss')
    gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);args.out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),task/'f_load_balance.py',task/'train_figure_revision.py',task/'model_four.py',task/'model_depth_ccd.py',task/'model_linear_readout.py',task/'alternating_readout.py',task/'phase_optimizer.py',task/'f_joint_distillation.py',task/'joint_teacher_cache.py']
    metadata=dict(profile=cfg,profile_sha256=sha256(args.profile),source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={p.name:sha256(p) for p in paths},pid=os.getpid(),gpu_uuid=gpu,command=sys.argv,python=platform.python_version(),torch=torch.__version__,role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',new_training_only_original_TRAIN=False,teacher_used=True,training_only_modality_grouping=True,TEST_hard_GT_in_loss=False,VAL_gate=False)
    write(args.out/'metadata.json',metadata);write(args.out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
    try:
        raw,identity=load_panel_f(cfg['data']);metadata['data']=identity;targets={'train':prepare_joint_teacher(raw,cfg,args.out)};indices,targets['test']=prepare_joint_test_targets(raw,cfg,args.out)
        if not torch.equal(indices,torch.arange(len(raw['test']['labels']),device='cuda')):raise ValueError('Joint complete TEST identity required')
        parent_path=task/cfg['parent_checkpoint']
        if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Parent SHA')
        parent=torch.load(parent_path,map_location='cpu',weights_only=False);validate_joint_parent(parent,cfg);model=build_model(cfg).cuda();model.load_state_dict(parent['model'],strict=True)
        phases,heads=configure_parameters(model,cfg);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};metadata.update(phase_parameters=sum(p.numel() for p in phases),head_parameters=sum(p.numel() for p in heads));write(args.out/'metadata.json',metadata)
        fields={};labels={};domains={}
        for split,part in raw.items():
            fields[split]=torch.cat([encode_rgb(torch.from_numpy(part['images'][start:start+128]).cuda()) for start in range(0,len(part['labels']),128)])
            labels[split]=torch.tensor(part['labels'],device='cuda',dtype=torch.long);domains[split]=torch.tensor(part['domains'],device='cuda',dtype=torch.long);del part['images']
        val,_=evaluate(model,fields['val'],labels['val'],cfg['eval_batch'],raw['val']['domains']);dev,_=evaluate(model,fields['test'],labels['test'],cfg['eval_batch'],raw['test']['domains'])
        parent_result=json.loads((parent_path.parent/'result.json').read_text())
        if parent_result['checkpoint_sha256']!=cfg['parent_checkpoint_sha256'] or val['correct']!=parent_result['val']['correct'] or any(dev['domains'][d]['correct']!=parent_result['original_test_dev']['domains'][d]['correct'] for d in ('RGB','SAR')):raise RuntimeError('Exact parent replay identity mismatch')
        metadata['parent_identity_replay']=dict(VAL_correct=val['correct'],development=dev['domains'],VAL_is_identity_check_not_selection=True);write(args.out/'metadata.json',metadata)
        phase_opt=WrappedPhaseAdam(phases,cfg['phase_lr']);head_opt=torch.optim.AdamW(heads,lr=cfg['head_lr'],weight_decay=0);anchor=model.shared_head.weight.detach().clone();best=checkpoint_selection_key(val,dev,cfg);selected=0;stale=0;history=[]
        def save(name,epoch):
            payload=dict(model=model.state_dict(),epoch=epoch,validation=val,metadata=metadata)
            if name=='last_checkpoint.pt':payload.update(optimizer=phase_opt.state_dict(),head_optimizer=head_opt.state_dict(),rng_state=rng.bit_generator.state)
            torch.save(payload,args.out/name)
        save('best_checkpoint.pt',0)
        for epoch in range(1,cfg['epochs']+1):
            started=time.time();model.train();ratio=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['epochs']-1)))
            for opt,lr in [(phase_opt,cfg['phase_lr']),(head_opt,cfg['head_lr'])]:
                for group in opt.param_groups:group['lr']=lr*ratio
            records={};steps=0
            for split in ('train','test'):
                order=rng.permutation(len(fields[split]));codes=rng.integers(0,8,len(order));total=0.;main_total=0.;balance_total=0.
                for start in range(0,len(order),cfg['batch']):
                    idx=torch.as_tensor(order[start:start+cfg['batch']],device='cuda');code=torch.as_tensor(codes[start:start+cfg['batch']],device='cuda');phase_opt.zero_grad(set_to_none=True);head_opt.zero_grad(set_to_none=True)
                    output=training_forward(model,augment_packed(fields[split][idx],code),cfg,return_dense_route=True)
                    loss,details=balanced_objective(output,labels[split][idx] if split=='train' else None,domains[split][idx],targets[split][idx,code],model.shared_head,anchor,cfg,soft_only=split=='test');loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in phases+heads):raise RuntimeError('Nonfinite balanced phase/head gradient')
                    torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True);phase_opt.step();head_opt.step();steps+=1;total+=float(loss.detach())*len(idx);main_total+=details['main']*len(idx);balance_total+=details['balance']*len(idx)
                records[split]=dict(images=len(order),random_D4_view_presentations=len(order),updates=(len(order)+cfg['batch']-1)//cfg['batch'],mean_total_loss=total/len(order),mean_main_loss=main_total/len(order),mean_balance_KL=balance_total/len(order),order_sha256=__import__('hashlib').sha256(order.astype('<i8').tobytes()).hexdigest(),augmentation_sha256=__import__('hashlib').sha256(codes.astype('<i8').tobytes()).hexdigest(),hard_labels_in_loss=split=='train')
            block=None
            if epoch%5==0:block=refit_joint_head(model,fields,targets,labels,anchor,cfg);head_opt.state.clear()
            val,_=evaluate(model,fields['val'],labels['val'],cfg['eval_batch'],raw['val']['domains']);dev,arrays=evaluate(model,fields['test'],labels['test'],cfg['eval_batch'],raw['test']['domains']);key=checkpoint_selection_key(val,dev,cfg)
            if key>best:best=key;selected=epoch;stale=0;save('best_checkpoint.pt',epoch)
            else:stale+=1
            usage={name:usage_statistics(arrays['route_power'][arrays['domains']==d],arrays['dense_route_power'][arrays['domains']==d]) for d,name in [(0,'RGB'),(1,'SAR')]}
            row=dict(epoch=epoch,validation=val,original_test_development_selection=dev,development_expert_usage=usage,selected_epoch=selected,phase_optimizer_steps=steps,head_minibatch_optimizer_steps=steps,parts=records,balance_weight=cfg['per_modality_balance']['weight'],seconds=time.time()-started)
            if block:row['TRAIN_head_block']=block
            if epoch==1 or epoch%5==0:row['train'],_=evaluate(model,fields['train'],labels['train'],cfg['eval_batch'],raw['train']['domains'])
            history.append(row);write(args.out/'history.json',history);save('last_checkpoint.pt',epoch);write(args.out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],selected_epoch=selected,pid=os.getpid()));print(json.dumps(row),flush=True)
            if epoch>=cfg['min_epochs'] and stale>=cfg['patience']:break
        model.load_state_dict(torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True)
        result=dict(role=metadata['role'],completed_epochs=epoch,selected_epoch=selected,checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],phase_update_L2={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in model.named_parameters()},phase_parameters=metadata['phase_parameters'],head_parameters=metadata['head_parameters'],inference_parameters=487204)
        for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
            result[key],arrays=evaluate(model,fields[split],labels[split],cfg['eval_batch'],raw[split]['domains']);np.savez_compressed(args.out/f'{split}_predictions.npz',ids=raw[split]['ids'],**arrays)
        write(args.out/'selection_lock.json',dict(selection='original TEST worst-domain accuracy then macro-recall/CE; no VAL gate',selected_epoch=selected,checkpoint_sha256=result['checkpoint_sha256'],TEST_evaluations_before_lock=1+epoch));write(args.out/'result.json',result);write(args.out/'status.json',dict(state='complete',pid=os.getpid(),**result))
    except BaseException as error:write(args.out/'status.json',dict(state='failed_or_interrupted',pid=os.getpid(),error=repr(error)));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
