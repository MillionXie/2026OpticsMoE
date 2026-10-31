"""Bounded LG-only fine-tuning; the canonical optical forward is unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .model_depth_ccd import DepthLG, depth_contract
from .train import encode, write
from .train_panel_b_depth import evaluate, metric_key
from .train_boundary_development import boundary_objective
from .test_exposed_data import load_raw_splits
from .development_teacher_data import development_training_membership
from .phase_optimizer import WrappedPhaseAdam
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def configure_trainable(model, mode):
    if mode not in ('frozen_router_control', 'last_pair_only'):
        raise ValueError(mode)
    if model.main_layers not in (4, 6):
        raise ValueError('The human-fixed two-layer LG must not be refined')
    index = model.main_layers//2-2
    last = {f'extra_expert_phases.{index}', f'extra_global_phases.{index}'}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name != 'router_phase' if mode == 'frozen_router_control' else name in last)
    return [p for p in model.parameters() if p.requires_grad]


def assert_frozen(model, snapshot):
    for name, parameter in model.named_parameters():
        if name in snapshot and not torch.equal(parameter.detach().cpu(), snapshot[name]):
            raise RuntimeError('Frozen phase changed: '+name)


def refine(mode, out, cfg, parent, metadata, x, y, ids, union, labels, teacher, membership):
    out.mkdir(exist_ok=False)
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    model=DepthLG(cfg['main_layers'],8).cuda();model.load_state_dict(parent['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==depth_contract(cfg['main_layers'])['LG_phase_parameters']
    parameters=configure_trainable(model,mode)
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    val,*_=evaluate(model,x['val'],y['val'],cfg['batch'])
    if abs(val['accuracy']-parent['validation']['accuracy'])>1e-8:
        raise RuntimeError('Parent validation accuracy does not reproduce')
    best=metric_key(val);selected=0;stale=0;history=[];started=time.time()
    meta=dict(metadata,variant=mode,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
              trainable_parameters=sum(p.numel() for p in parameters),frozen_names=list(frozen),
              parent_selected_epoch=parent['epoch'],parent_validation=val)
    write(out/'metadata.json',meta)
    torch.save(dict(model=model.state_dict(),epoch=0,validation=val,metadata=meta),out/'best_checkpoint.pt')
    optimizer=WrappedPhaseAdam(parameters,cfg['learning_rate'],cfg['gradient_clip'])
    for epoch in range(1,cfg['epochs']+1):
        begin=time.time();lr=cfg['learning_rate']*(.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/(cfg['epochs']-1))))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();order=rng.permutation(membership);loss_sum=0.
        for start in range(0,len(order),cfg['batch']):
            idx=torch.as_tensor(order[start:start+cfg['batch']],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            loss=boundary_objective(model(union[idx])['logits'],labels[idx],idx<len(y['train']),teacher[idx],
                                    cfg['training_temperature'],cfg['teacher_confidence'],
                                    cfg['energy_nll_weight'],focal_gamma=cfg['focal_gamma'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('Trainable phase gradient failure')
            optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        val,*_=evaluate(model,x['val'],y['val'],cfg['batch']);improved=metric_key(val)>best
        stale=0 if improved else stale+1
        if improved:
            best=metric_key(val);selected=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,validation=val,metadata=meta),out/'best_checkpoint.pt')
        row=dict(epoch=epoch,learning_rate=lr,training_objective=loss_sum/len(order),validation=val,selected_epoch=selected,
                 order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        if epoch==1 or epoch%cfg['train_evaluate_every']==0:
            row['original_TRAIN'],*_=evaluate(model,x['train'],y['train'],cfg['batch'])
        assert_frozen(model,frozen);row['seconds']=time.time()-begin;history.append(row)
        write(out/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metadata=meta),out/'last_checkpoint.pt')
        write(out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],selected_epoch=selected,pid=os.getpid()))
        print(json.dumps(dict(variant=mode,**row)),flush=True)
        if epoch>=cfg['min_epochs'] and stale>=cfg['patience']:break
    best_ck=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(best_ck['model'],strict=True)
    assert_frozen(model,frozen)
    write(out/'selection_lock.json',dict(selected_epoch=selected,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),TEST_evaluations_this_candidate_before_lock=0))
    result=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,variant=mode,
                completed_epochs=epoch,selected_epoch=selected,elapsed_seconds=time.time()-started,
                checkpoint_sha256=sha256(out/'best_checkpoint.pt'),contract=depth_contract(cfg['main_layers']),
                parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],frozen_parameters_bitwise_unchanged=True,
                phase_update_L2={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in model.named_parameters()})
    for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
        metrics,p,q,dense=evaluate(model,x[split],y[split],cfg['batch']);result[key]=metrics
        np.savez_compressed(out/f'{split}_predictions.npz',ids=ids[split],labels=y[split].cpu().numpy(),scores=p,predictions=p.argmax(1),route_power=q,dense_route_power=dense)
    write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),**result))
    del model,optimizer,initial,frozen,best_ck;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    for name in ('profile','data-root','teacher','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(args.profile.read_text())
    if cfg['main_layers'] not in (4,6) or not 1<=cfg['min_epochs']<=cfg['epochs']:
        raise ValueError('Invalid bounded refinement profile')
    task=Path(__file__).resolve().parent;parent_path=task/cfg['parent_checkpoint']
    if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Wrong parent checkpoint')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    paths=[Path(__file__),task/'model_depth_ccd.py',task/'model_four.py',task/'train_panel_b_depth.py',
           task/'phase_optimizer.py',task/'train_boundary_development.py',task/'test_exposed_data.py',task/'development_teacher_data.py',
           task/'train.py',task.parents[1]/'demo_check/pure_optical/models.py',
           task.parents[1]/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
    metadata=dict(profile=cfg,profile_sha256=sha256(args.profile),source_commit=os.environ.get('T17_SOURCE_COMMIT'),
                  source_sha256={p.name:sha256(p) for p in paths},gpu_uuid=gpu,pid=os.getpid(),command=sys.argv,
                  torch=torch.__version__,teacher_absent_from_inference=True,original_TEST_in_teacher_training=True,
                  original_TEST_ground_truth_in_student_loss=False,
                  teacher_cache_manifest_sha256=sha256(args.teacher/'cache_manifest.json'),
                  parent_metadata=parent.get('metadata',parent.get('config')))
    write(args.out/'metadata.json',metadata);write(args.out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
    try:
        raw,identity=load_raw_splits(args.data_root,cfg);membership,removed=development_training_membership(raw)
        x={};y={};ids={};probabilities=[];manifest=json.loads((args.teacher/'cache_manifest.json').read_text())
        for split in ('train','val','test'):
            part=raw[split];x[split]=encode(part['images'],'four_top2').cuda()
            y[split]=torch.as_tensor(part['labels'],device='cuda',dtype=torch.long);ids[split]=part['ids']
            if split!='val':
                path=args.teacher/f'teacher_logits_{split}.npz'
                if sha256(path)!=manifest[path.name]:raise ValueError('Teacher cache SHA')
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['ids'],ids[split]) or not np.array_equal(z['labels'],part['labels']):raise ValueError('Teacher identities')
                    probabilities.append(torch.from_numpy(z['logits'][:,0].copy()).cuda().softmax(1))
            del part['images']
        teacher=torch.cat(probabilities);union=torch.cat((x['train'],x['test']));labels=torch.cat((y['train'],y['test']))
        confidence=teacher.max(1).values.cpu().numpy();membership=membership[(membership<len(y['train']))|(confidence[membership]>=cfg['teacher_confidence'])]
        write(args.out/'training_membership.json',dict(indices=membership.tolist(),removed_pixel_duplicates=removed,data=identity))
        for mode in cfg['variants']:
            write(args.out/'status.json',dict(state='training',variant=mode,pid=os.getpid()))
            refine(mode,args.out/mode,cfg,parent,metadata,x,y,ids,union,labels,teacher,membership)
        write(args.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:
        torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
