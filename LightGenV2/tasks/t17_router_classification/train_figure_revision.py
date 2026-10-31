"""Bounded B Linear adaptation and original F RGB/SAR rerun, explicit GPU allowlist."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model_linear_readout import LinearReadoutLG
from .model_depth_ccd import DepthD2NN
from .phase_optimizer import WrappedPhaseAdam
from .gpu_policy import require_authorized_gpu
from .readout_data import load_raw
from .figure_revision_data import load_panel_f
from .train import write
from .model_four import encode_rgb
from .alternating_readout import domain_route_loss, anchored_classification, refit_existing_head
from .train_optical_four_ccd import route_statistics
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def build_model(cfg):
    if cfg['architecture']=='LG':return LinearReadoutLG(cfg['main_layers'],cfg['num_classes'])
    if cfg['architecture']=='D2NN':return DepthD2NN(cfg['layers'],cfg['num_classes'])
    raise ValueError(cfg['architecture'])


def configure_parameters(model,cfg):
    scope=cfg.get('training_scope','all')
    if scope not in ('all','last_pair_and_head','last_pair_fixed_head','all_optics_fixed_head'):raise ValueError('Unknown training scope')
    if scope in ('last_pair_and_head','last_pair_fixed_head'):
        if cfg['architecture']!='LG' or cfg['main_layers'] not in (4,6):raise ValueError('Last pair scope requires deep LG')
        last=cfg['main_layers']//2-2
        names={f'extra_expert_phases.{last}',f'extra_global_phases.{last}'}
        if scope=='last_pair_and_head':names.add('shared_head.weight')
        for name,p in model.named_parameters():p.requires_grad_(name in names)
    elif scope=='all_optics_fixed_head':
        if cfg['architecture']!='LG':raise ValueError('Fixed head scope requires LG')
        for name,p in model.named_parameters():p.requires_grad_(name!='shared_head.weight')
    else:
        for p in model.parameters():p.requires_grad_(True)
    phases=[p for n,p in model.named_parameters() if p.requires_grad and n!='shared_head.weight']
    head=[p for n,p in model.named_parameters() if p.requires_grad and n=='shared_head.weight']
    return phases,head


def augment_packed(x,codes):
    """Apply geometry within each channel tile, never rotate RGB tile positions."""
    if x.shape[-2:]!=(224,224):raise ValueError('Packed224 expected')
    tiles=torch.stack([x[:,:112,:112],x[:,:112,112:],x[:,112:,:112],x[:,112:,112:]],1)
    transformed=torch.empty_like(tiles)
    for code in range(8):
        mask=codes==code
        if not bool(mask.any()):continue
        value=torch.rot90(tiles[mask],code%4,(-2,-1))
        if code>=4:value=value.flip(-1)
        transformed[mask]=value
    return torch.cat([torch.cat([transformed[:,0],transformed[:,1]],-1),torch.cat([transformed[:,2],transformed[:,3]],-1)],-2)


@torch.no_grad()
def evaluate(model,x,labels,batch,domains=None):
    model.eval();scores=[];routes=[];dense=[];capture=[]
    for start in range(0,len(labels),batch):
        part=x[start:start+batch];out=model(part,return_dense_route=True) if isinstance(model,LinearReadoutLG) else model(part)
        scores.append(out['probabilities'].cpu().numpy())
        if out.get('route_power') is not None:
            routes.append(out['route_power'].cpu().numpy());dense.append(out['dense_route_power'].cpu().numpy());capture.append(out['router_capture'].cpu().numpy())
    scores=np.concatenate(scores);truth=labels.cpu().numpy();pred=scores.argmax(1);classes=scores.shape[1]
    cm=np.bincount(truth*classes+pred,minlength=classes**2).reshape(classes,classes)
    metrics=dict(n=len(truth),correct=int((pred==truth).sum()),accuracy=float((pred==truth).mean()),
        balanced_accuracy=float((cm.diagonal()/cm.sum(1).clip(1)).mean()),
        cross_entropy=float(-np.log(scores[np.arange(len(truth)),truth].clip(1e-30)).mean()),confusion_matrix=cm.tolist())
    arrays=dict(scores=scores,predictions=pred,labels=truth)
    if routes:
        arrays.update(route_power=np.concatenate(routes),dense_route_power=np.concatenate(dense),router_capture=np.concatenate(capture));metrics['router']=route_statistics(arrays['route_power'])
    if domains is not None:
        metrics['domains']={}
        for domain,name in [(0,'RGB'),(1,'SAR')]:
            mask=domains==domain;metrics['domains'][name]=dict(n=int(mask.sum()),correct=int((pred[mask]==truth[mask]).sum()),accuracy=float((pred[mask]==truth[mask]).mean()))
            if routes:metrics['domains'][name]['router']=route_statistics(arrays['route_power'][mask])
        metrics['domain_mean_accuracy']=np.mean([m['accuracy'] for m in metrics['domains'].values()]).item()
        arrays['domains']=domains
    return metrics,arrays


def metric_key(metrics):return metrics['balanced_accuracy'],-metrics['cross_entropy']


def classification_objective(output,labels,cfg):
    # No teacher, TEST labels, domain labels, routing oracle, or auxiliary head.
    logits=output['raw_logits'] if cfg['architecture']=='LG' else output['logits']/cfg['training_temperature']
    return F.cross_entropy(logits,labels,label_smoothing=cfg['label_smoothing'])


def assert_frozen(model,snapshot):
    for name,p in model.named_parameters():
        if name in snapshot and not torch.equal(p.detach().cpu(),snapshot[name]):raise RuntimeError('Frozen tensor changed: '+name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);task=Path(__file__).resolve().parent
    args.out.mkdir(exist_ok=False,parents=True)
    paths=[Path(__file__),task/'alternating_readout.py',task/'train_convex_readout.py',task/'model_linear_readout.py',task/'model_depth_ccd.py',task/'model_four.py',task/'model_linear_ccd.py',task/'gpu_policy.py',task/'figure_revision_data.py',task/'phase_optimizer.py',task/'readout_data.py',task.parents[1]/'demo_check/pure_optical/models.py',task.parents[1]/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
    metadata=dict(profile=cfg,profile_sha256=sha256(args.profile),source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={p.name:sha256(p) for p in paths},gpu_uuid=gpu,pid=os.getpid(),command=sys.argv,python=platform.python_version(),torch=torch.__version__,role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',new_training_only_original_TRAIN=True,teacher_used=False)
    write(args.out/'metadata.json',metadata);write(args.out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
    try:
        raw,identity=load_panel_f(cfg['data']) if cfg['dataset']=='eurosat_original_f' else load_raw(Path(cfg['data_root']),cfg)
        metadata['data']=identity;model=build_model(cfg).cuda();parent=None
        if cfg.get('parent_checkpoint'):
            parent_path=task/cfg['parent_checkpoint']
            if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Parent SHA differs')
            parent=torch.load(parent_path,map_location='cpu',weights_only=False);model.load_state_dict(parent['model'],strict=True)
        phases,heads=configure_parameters(model,cfg);frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        metadata.update(trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],phase_parameters=sum(p.numel() for n,p in model.named_parameters() if n!='shared_head.weight'),head_parameters=sum(p.numel() for n,p in model.named_parameters() if n=='shared_head.weight'))
        write(args.out/'metadata.json',metadata);x={};y={}
        for split,part in raw.items():
            fields=[]
            for start in range(0,len(part['labels']),128):fields.append(encode_rgb(torch.from_numpy(part['images'][start:start+128]).cuda()))
            x[split]=torch.cat(fields);y[split]=torch.as_tensor(part['labels'],device='cuda',dtype=torch.long);del part['images']
        block=cfg.get('alternating_head');aux=cfg.get('domain_route_aux',0.)
        if (block or aux) and (cfg['architecture']!='LG' or cfg['dataset']!='eurosat_original_f' or not heads):raise ValueError('Alternating/domain method requires original F LG with trainable head')
        anchor=model.shared_head.weight.detach().clone() if block else None
        phase_opt=WrappedPhaseAdam(phases,cfg['phase_lr']) if cfg['phase_optimizer']=='wrapped' else torch.optim.Adam(phases,lr=cfg['phase_lr'])
        head_opt=torch.optim.AdamW(heads,lr=cfg['head_lr'],weight_decay=cfg['head_weight_decay']) if heads else None
        val,_=evaluate(model,x['val'],y['val'],cfg['eval_batch'],raw['val'].get('domains'));selected=0;best=metric_key(val);stale=0;history=[]
        if parent is not None and abs(val['accuracy']-parent['validation']['accuracy'])>1e-7:raise RuntimeError('Exact parent VAL does not reproduce')
        torch.save(dict(model=model.state_dict(),epoch=0,validation=val,metadata=metadata),args.out/'best_checkpoint.pt')
        for epoch in range(1,cfg['epochs']+1):
            started=time.time();model.train();order=rng.permutation(len(y['train']));codes=rng.integers(0,8,len(order)) if cfg['augmentation']=='D4' else rng.integers(0,2,len(order))*4 if cfg['augmentation']=='horizontal_flip' else np.zeros(len(order),dtype=int)
            ratio=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['epochs']-1))) if cfg['cosine_lr'] else 1.
            for optimizer,lr in [(phase_opt,cfg['phase_lr']),(head_opt,cfg['head_lr'])]:
                if optimizer:
                    for group in optimizer.param_groups:group['lr']=lr*ratio
            loss_sum=0.
            for start in range(0,len(order),cfg['batch']):
                idx=torch.as_tensor(order[start:start+cfg['batch']],device='cuda');code=torch.as_tensor(codes[start:start+cfg['batch']],device='cuda');field=augment_packed(x['train'][idx],code)
                phase_opt.zero_grad(set_to_none=True)
                if head_opt:head_opt.zero_grad(set_to_none=True)
                output=model(field,return_dense_route=True) if aux else model(field)
                loss=anchored_classification(output['raw_logits'],y['train'][idx],model.shared_head.weight,anchor,block['anchor_strength'],cfg['label_smoothing']) if block else classification_objective(output,y['train'][idx],cfg)
                if aux:
                    acquisition=torch.as_tensor(raw['train']['domains'][order[start:start+cfg['batch']]],device='cuda',dtype=torch.long)
                    loss=loss+aux*domain_route_loss(output['dense_route_power'],acquisition)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                for p in phases+heads:
                    if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Missing/nonfinite trainable gradient')
                if cfg['phase_optimizer']!='wrapped':torch.nn.utils.clip_grad_norm_(phases,5.,error_if_nonfinite=True)
                if heads:torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True)
                phase_opt.step()
                if head_opt:head_opt.step()
                loss_sum+=float(loss.detach())*len(idx)
            block_record=None
            if block and epoch%block['interval_epochs']==0:
                model.eval();features=[]
                with torch.no_grad():
                    for start in range(0,len(y['train']),cfg['eval_batch']):features.append(model.optical_features(x['train'][start:start+cfg['eval_batch']])['ccd_features'])
                block_record=refit_existing_head(model.shared_head,torch.cat(features),y['train'],anchor,block['anchor_strength'],cfg['label_smoothing'],block['inner_steps'])
                head_opt.state.clear() # Momentum belongs to the old head coordinates.
            val,_=evaluate(model,x['val'],y['val'],cfg['eval_batch'],raw['val'].get('domains'));improved=metric_key(val)>best;stale=0 if improved else stale+1
            if improved:
                selected=epoch;best=metric_key(val);torch.save(dict(model=model.state_dict(),epoch=epoch,validation=val,metadata=metadata),args.out/'best_checkpoint.pt')
            row=dict(epoch=epoch,phase_lr=cfg['phase_lr']*ratio,head_lr=cfg['head_lr']*ratio,training_objective=loss_sum/len(order),validation=val,selected_epoch=selected,order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),augmentation_codes_sha256=hashlib.sha256(codes.astype('<i8').tobytes()).hexdigest())
            if block_record:row['TRAIN_head_block']=block_record
            if epoch==1 or epoch%5==0:row['train'],_=evaluate(model,x['train'],y['train'],cfg['eval_batch'],raw['train'].get('domains'))
            assert_frozen(model,frozen);row['seconds']=time.time()-started;history.append(row);write(args.out/'history.json',history)
            torch.save(dict(model=model.state_dict(),epoch=epoch,optimizer=phase_opt.state_dict(),head_optimizer=head_opt.state_dict() if head_opt else None,metadata=metadata),args.out/'last_checkpoint.pt')
            write(args.out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],selected_epoch=selected,pid=os.getpid()));print(json.dumps(row),flush=True)
            if cfg['early_stop'] and epoch>=cfg['min_epochs'] and stale>=cfg['patience']:break
        selected_ck=torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected_ck['model'],strict=True);assert_frozen(model,frozen)
        write(args.out/'selection_lock.json',dict(selected_epoch=selected,checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),selection='VAL macro recall then CE',TEST_evaluations_this_run_before_lock=0))
        result=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,completed_epochs=epoch,selected_epoch=selected,checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),frozen_parameters_bitwise_unchanged=True,phase_update_L2={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in model.named_parameters()},phase_parameters=metadata['phase_parameters'],head_parameters=metadata['head_parameters'])
        for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
            metrics,arrays=evaluate(model,x[split],y[split],cfg['eval_batch'],raw[split].get('domains'));result[key]=metrics
            np.savez_compressed(args.out/f'{split}_predictions.npz',ids=raw[split]['ids'],**arrays)
        write(args.out/'result.json',result);write(args.out/'status.json',dict(state='complete',pid=os.getpid(),**result))
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',pid=os.getpid(),error=repr(error)));raise
    finally:
        torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
