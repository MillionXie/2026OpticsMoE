"""TRAIN-only convex fit of the existing Linear, with all optical phases frozen."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .gpu_policy import require_authorized_gpu
from .model_linear_readout import LinearReadoutLG
from .model_four import encode_rgb
from .readout_data import load_raw
from .figure_revision_data import load_panel_f
from .train_linear_readout import measure,key
from .train import write
from .train_optical_four_ccd import route_statistics
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def convex_objective(weight,features,labels,anchor,strength):
    if strength<=0:raise ValueError('positive anchor strength required')
    return F.cross_entropy(F.linear(features,weight),labels)+.5*strength*(weight-anchor).square().sum()


@torch.no_grad()
def cache(model,part,path,batch):
    values={k:[] for k in ['ccd_features','route_power','dense_route_power','router_capture']}
    for start in range(0,len(part['labels']),batch):
        field=encode_rgb(torch.from_numpy(part['images'][start:start+batch]).cuda())
        out=model.optical_features(field,return_dense_route=True)
        for k in values:values[k].append(out[k].cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in values.items()};arrays.update(ids=part['ids'],labels=part['labels'])
    if 'domains' in part:arrays['domains']=part['domains']
    if not all(np.isfinite(arrays[k]).all() for k in values):raise RuntimeError('Nonfinite frozen optical cache')
    np.savez_compressed(path,**arrays);return arrays


def run_job(name,cfg,common,out):
    task=Path(__file__).resolve().parent;out.mkdir(exist_ok=False);parent_path=task/cfg['parent_checkpoint']
    if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Wrong parent')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);model=LinearReadoutLG(cfg['main_layers'],cfg['num_classes']).cuda();model.load_state_dict(parent['model'],strict=True);model.requires_grad_(False)
    metadata=dict(job=name,profile=cfg,optimizer=common,role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={p.name:sha256(p) for p in [Path(__file__),task/'model_linear_readout.py',task/'model_four.py',task/'model_depth_ccd.py',task/'train_linear_readout.py',task/'figure_revision_data.py',task/'readout_data.py',task.parents[1]/'demo_check/pure_optical/models.py',task.parents[1]/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']},pid=os.getpid(),command=sys.argv,python=platform.python_version(),torch=torch.__version__,new_training_data='TRAIN labels only',TEST_used_in_fit=False,teacher_used=False,optimization_coordinates='float64 Linear weights only; export float32 canonical head before every selection',inference_parameter_count=sum(p.numel() for p in model.parameters()))
    write(out/'metadata.json',metadata);write(out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
    raw,identity=load_panel_f(cfg['data']) if cfg['dataset']=='eurosat_original_f' else load_raw(Path(cfg['data_root']),cfg)
    metadata['data']=identity;write(out/'metadata.json',metadata);arrays={};x={};y={}
    for split,part in raw.items():
        write(out/'status.json',dict(state='caching_frozen_optics',split=split,pid=os.getpid()))
        arrays[split]=cache(model,part,out/f'{split}_features.npz',common['optical_batch']);x[split]=torch.from_numpy(arrays[split]['ccd_features']).cuda();y[split]=torch.tensor(part['labels'],device='cuda',dtype=torch.long)
    write(out/'cache_manifest.json',dict(parent_checkpoint_sha256=sha256(parent_path),source_sha256=metadata['source_sha256'],feature_sha256={s:sha256(out/f'{s}_features.npz') for s in arrays},all_original_IDs_preserved=True,TEST_labels_absent_from_optimizer=True))
    original=model.shared_head.weight.detach().double().clone();weight=torch.nn.Parameter(original.clone());features=x['train'].double()
    optimizer=torch.optim.LBFGS([weight],lr=1.,max_iter=common['inner_steps'],max_eval=common['inner_steps']*2,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
    val,_=measure(model.shared_head,x['val'],y['val']);best=key(val);selected=0;history=[];evaluations=0
    torch.save(dict(model=model.state_dict(),epoch=0,validation=val,metadata=metadata),out/'best_checkpoint.pt')
    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True);loss=convex_objective(weight,features,y['train'],original,common['anchor_strength'])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite convex loss')
        loss.backward();evaluations+=1
        if not torch.isfinite(weight.grad).all():raise RuntimeError('Nonfinite head gradient')
        return loss
    for iteration in range(1,common['outer_steps']+1):
        start=time.time();optimizer.step(closure)
        with torch.no_grad():model.shared_head.weight.copy_(weight.float())
        train,_=measure(model.shared_head,x['train'],y['train']);val,_=measure(model.shared_head,x['val'],y['val'])
        if key(val)>best:
            best=key(val);selected=iteration;torch.save(dict(model=model.state_dict(),epoch=iteration,validation=val,metadata=metadata),out/'best_checkpoint.pt')
        objective=closure();state=optimizer.state[weight]
        row=dict(iteration=iteration,training=train,validation=val,selected_iteration=selected,regularized_training_objective=float(objective.detach()),gradient_max_abs=float(weight.grad.abs().max()),lbfgs_inner_iterations=state.get('n_iter',0),function_evaluations=evaluations,seconds=time.time()-start);history.append(row);write(out/'history.json',history)
        torch.save(dict(model=model.state_dict(),epoch=iteration,optimizer=optimizer.state_dict(),metadata=metadata),out/'last_checkpoint.pt');write(out/'status.json',dict(state='fitting_convex_Linear',iteration=iteration,total_iterations=common['outer_steps'],pid=os.getpid()));print(json.dumps(row),flush=True)
        if row['gradient_max_abs']<common['stop_gradient_max_abs']:break
    ck=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    for n,v in parent['model'].items():
        if n!='shared_head.weight' and not torch.equal(model.state_dict()[n].cpu(),v):raise RuntimeError('Frozen optical tensor changed')
    write(out/'selection_lock.json',dict(selected_iteration=selected,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),selection='VAL macro recall then CE, parent E0 participates',TEST_evaluations_before_lock=0))
    result=dict(job=name,role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,completed_outer_steps=iteration,completed_inner_steps=state.get('n_iter',0),function_evaluations=evaluations,selected_epoch=selected,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),optical_phases_bitwise_parent=True,final_training_head_fit=history[-1]['training'],final_training_gradient_max_abs=history[-1]['gradient_max_abs'],parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'])
    for split,label in [('train','train'),('val','val'),('test','original_test_dev')]:
        metrics,scores=measure(model.shared_head,x[split],y[split]);a=arrays[split];metrics['router']=route_statistics(a['route_power'])
        if 'domains' in a:
            pred=scores.argmax(1);metrics['domains']={title:dict(n=int((a['domains']==d).sum()),correct=int((pred[a['domains']==d]==a['labels'][a['domains']==d]).sum()),accuracy=float((pred[a['domains']==d]==a['labels'][a['domains']==d]).mean())) for d,title in [(0,'RGB'),(1,'SAR')]}
        result[label]=metrics;np.savez_compressed(out/f'{split}_predictions.npz',ids=a['ids'],labels=a['labels'],scores=scores,predictions=scores.argmax(1),route_power=a['route_power'],dense_route_power=a['dense_route_power'],router_capture=a['router_capture'],**({'domains':a['domains']} if 'domains' in a else {}))
    with torch.no_grad():
        replay=[]
        for start in range(0,len(raw['test']['labels']),common['optical_batch']):replay.append(model(encode_rgb(torch.from_numpy(raw['test']['images'][start:start+common['optical_batch']]).cuda()))['probabilities'].cpu().numpy())
    replay=np.concatenate(replay);cached=np.load(out/'test_predictions.npz')['scores'];assert np.array_equal(replay.argmax(1),cached.argmax(1)) and np.allclose(replay,cached,atol=3e-6,rtol=1e-4)
    result['canonical_full_TEST_replay']=dict(predictions_identical=True,max_probability_difference=float(np.max(np.abs(replay-cached))))
    write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),**result));del model,weight,optimizer,features,arrays,x,y,raw,parent,ck;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);args.out.mkdir(exist_ok=False,parents=True)
    write(args.out/'launch_context.json',dict(source_commit=os.environ.get('T17_SOURCE_COMMIT'),gpu_uuid=gpu,pid=os.getpid(),profile_sha256=sha256(args.profile),command=sys.argv))
    try:
        for name,job in cfg['jobs'].items():run_job(name,job,cfg['optimization'],args.out/name)
        write(args.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
