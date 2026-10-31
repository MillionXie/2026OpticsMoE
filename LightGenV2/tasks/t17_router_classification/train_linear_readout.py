"""Fit one Linear on frozen, SHA-bound optical features; TRAIN labels only."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model_linear_readout import LinearReadoutLG,load_optical_parent
from .model_depth_ccd import depth_contract
from .readout_data import load_raw
from .train import encode,write
from .train_optical_four_ccd import route_statistics
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


@torch.no_grad()
def measure(head,x,y):
    scores=head(x).softmax(1);pred=scores.argmax(1);classes=scores.shape[1]
    cm=torch.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes)
    return dict(n=len(y),correct=int(cm.diag().sum()),accuracy=float((pred==y).float().mean()),
                balanced_accuracy=float((cm.diag()/cm.sum(1).clamp_min(1)).mean()),
                cross_entropy=float(F.nll_loss(scores.clamp_min(1e-30).log(),y)),confusion_matrix=cm.cpu().tolist()),scores.cpu().numpy()


def key(metrics):return metrics['balanced_accuracy'],-metrics['cross_entropy']


@torch.no_grad()
def cache_features(model,raw,out,batch):
    features=[];routes=[];dense=[];capture=[]
    for start in range(0,len(raw['labels']),batch):
        result=model.optical_features(encode(raw['images'][start:start+batch],'four_top2').cuda(),True)
        for accumulator,name in [(features,'ccd_features'),(routes,'route_power'),(dense,'dense_route_power'),(capture,'router_capture')]:
            accumulator.append(result[name].cpu().numpy())
    arrays=dict(ids=raw['ids'],labels=raw['labels'],features=np.concatenate(features),route_power=np.concatenate(routes),
                dense_route_power=np.concatenate(dense),router_capture=np.concatenate(capture))
    if not np.isfinite(arrays['features']).all():raise RuntimeError('Nonfinite optical feature cache')
    np.savez_compressed(out,**arrays)
    return arrays


def train_job(name,cfg,out,common):
    out.mkdir(exist_ok=False);torch.manual_seed(common['seed']);rng=np.random.default_rng(common['seed'])
    task=Path(__file__).resolve().parent;parent_path=task/cfg['parent_checkpoint']
    if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Optical parent SHA')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    model=LinearReadoutLG(cfg['main_layers'],cfg['num_classes']).cuda();load_optical_parent(model,parent['model'])
    for n,p in model.named_parameters():p.requires_grad_(n=='shared_head.weight')
    optical_count=sum(p.numel() for n,p in model.named_parameters() if n!='shared_head.weight')
    expected=depth_contract(cfg['main_layers'])['LG_phase_parameters']
    assert optical_count==expected and model.shared_head.weight.numel()==784*cfg['num_classes']
    paths=[Path(__file__),task/'model_linear_readout.py',task/'model_depth_ccd.py',task/'model_four.py',task/'readout_data.py',
           task/'train.py',task/'train_linear_ccd_8class.py',task/'test_exposed_data.py',task/'train_optical_four_ccd.py',
           task.parents[1]/'demo_check/pure_optical/models.py',
           task.parents[1]/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
    metadata=dict(job=name,profile=cfg,training=common,source_commit=os.environ.get('T17_SOURCE_COMMIT'),
                  source_sha256={p.name:sha256(p) for p in paths},command=sys.argv,pid=os.getpid(),torch=torch.__version__,python=platform.python_version(),
                  role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',optical_parent_already_TEST_exposed=True,
                  new_head_TRAIN_labels_only=True,new_teacher_or_TEST_training=False,
                  optical_phase_parameters=optical_count,electronic_head_parameters=784*cfg['num_classes'],
                  trainable_names=['shared_head.weight'],head='CCD active478 -> avgpool28 -> mean normalization -> Linear(784,C,bias=False)',
                  OEO='Unchanged inherited PhaseOnly.oeo after each main phase',parent_optical_SHA=cfg['parent_checkpoint_sha256'])
    write(out/'metadata.json',metadata);write(out/'status.json',dict(state='loading_verified_raw_data',pid=os.getpid()))
    raw,identity=load_raw(Path(cfg['data_root']),cfg);metadata['data']=identity;write(out/'metadata.json',metadata)
    cache={};x={};y={}
    for split in ('train','val','test'):
        write(out/'status.json',dict(state='caching_frozen_optics',split=split,pid=os.getpid()))
        cache[split]=cache_features(model,raw[split],out/f'{split}_features.npz',common['optical_batch'])
        x[split]=torch.from_numpy(cache[split]['features']).cuda();y[split]=torch.tensor(raw[split]['labels'],device='cuda',dtype=torch.long)
    write(out/'cache_manifest.json',dict(parent_sha256=cfg['parent_checkpoint_sha256'],
          features={s:sha256(out/f'{s}_features.npz') for s in cache},source_sha256=metadata['source_sha256'],
          TEST_features_cached_without_fitting=True,TEST_labels_absent_from_head_training=True))
    optimizer=torch.optim.AdamW(model.shared_head.parameters(),lr=common['learning_rate'],weight_decay=common['weight_decay'])
    baseline,_=measure(model.shared_head,x['val'],y['val']);best=key(baseline);selected=0;stale=0;history=[];started=time.time()
    torch.save(dict(model=model.state_dict(),epoch=0,validation=baseline,metadata=metadata),out/'best_checkpoint.pt')
    for epoch in range(1,common['epochs']+1):
        begin=time.time();model.shared_head.train();order=rng.permutation(len(y['train']));loss_sum=0.
        lr=common['learning_rate']*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/(common['epochs']-1))))
        for group in optimizer.param_groups:group['lr']=lr
        for start in range(0,len(order),common['batch']):
            idx=torch.as_tensor(order[start:start+common['batch']],device='cuda');optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model.shared_head(x['train'][idx]),y['train'][idx],label_smoothing=common['label_smoothing'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Linear loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.shared_head.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        val,_=measure(model.shared_head,x['val'],y['val']);improved=key(val)>best;stale=0 if improved else stale+1
        if improved:
            best=key(val);selected=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,validation=val,metadata=metadata),out/'best_checkpoint.pt')
        row=dict(epoch=epoch,lr=lr,train_objective=loss_sum/len(order),validation=val,selected_epoch=selected,
                 order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        if epoch==1 or epoch%5==0:row['training'],_=measure(model.shared_head,x['train'],y['train'])
        row['seconds']=time.time()-begin;history.append(row);write(out/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metadata=metadata),out/'last_checkpoint.pt')
        write(out/'status.json',dict(state='training_head_only',epoch=epoch,total_epochs=common['epochs'],selected_epoch=selected,pid=os.getpid()))
        print(json.dumps(dict(job=name,**row)),flush=True)
        if epoch>=common['min_epochs'] and stale>=common['patience']:break
    ck=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    for n,value in parent['model'].items():
        if not torch.equal(model.state_dict()[n].cpu(),value.cpu()):raise RuntimeError('Frozen optical parent changed: '+n)
    write(out/'selection_lock.json',dict(selected_epoch=selected,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),selection='VAL macro recall then cross entropy',TEST_evaluations_before_lock=0))
    result=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,job=name,completed_epochs=epoch,
                selected_epoch=selected,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),optical_parent_sha256=cfg['parent_checkpoint_sha256'],
                optical_phases_bitwise_frozen=True,optical_phase_parameters=optical_count,linear_head_parameters=784*cfg['num_classes'],
                elapsed_head_training_seconds=time.time()-started)
    for split,key_name in [('train','train'),('val','val'),('test','original_test_dev')]:
        metrics,scores=measure(model.shared_head,x[split],y[split]);result[key_name]=metrics
        metrics['router']=route_statistics(cache[split]['route_power'])
        np.savez_compressed(out/f'{split}_predictions.npz',ids=raw[split]['ids'],labels=raw[split]['labels'],scores=scores,
                            predictions=scores.argmax(1),route_power=cache[split]['route_power'],dense_route_power=cache[split]['dense_route_power'],
                            router_capture=cache[split]['router_capture'])
    full=[]
    with torch.no_grad():
        for start in range(0,len(raw['test']['labels']),common['optical_batch']):
            full.append(model(encode(raw['test']['images'][start:start+common['optical_batch']],'four_top2').cuda())['probabilities'].cpu().numpy())
    full=np.concatenate(full);cached_scores=np.load(out/'test_predictions.npz',allow_pickle=False)['scores']
    if not np.allclose(full,cached_scores,rtol=1e-4,atol=2e-6) or not np.array_equal(full.argmax(1),cached_scores.argmax(1)):
        raise RuntimeError('Full canonical-model TEST replay differs from feature cache')
    result['full_model_TEST_replay']=dict(all_samples=len(full),predictions_identical=True,max_probability_difference=float(np.max(np.abs(full-cached_scores))))
    write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),**result))
    del model,optimizer,parent,ck,raw,cache,x,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--jobs',nargs='+',required=True);args=parser.parse_args()
    gpu=require_non_a100_gpu();torch.set_num_threads(4);profile=json.loads(args.profile.read_text());args.out.mkdir(exist_ok=False,parents=True)
    write(args.out/'launch_context.json',dict(command=sys.argv,pid=os.getpid(),gpu_uuid=gpu,source_commit=os.environ.get('T17_SOURCE_COMMIT'),profile_sha256=sha256(args.profile)))
    try:
        for name in args.jobs:
            write(args.out/'status.json',dict(state='running',job=name,pid=os.getpid()))
            train_job(name,profile['jobs'][name],args.out/name,profile['training'])
        write(args.out/'status.json',dict(state='complete',jobs=args.jobs,pid=os.getpid()))
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:
        torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
