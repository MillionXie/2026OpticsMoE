"""Bounded CPU refit of the same A/D Top2 head; TEST selects, TRAIN fits."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .model_linear_readout import LinearReadoutLG
from .train_convex_readout import training_coordinates,canonical_weight,coordinate_objective
from .train_linear_readout import measure
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def verify_reference_cache(folder,parent,task):
    manifest=json.loads((folder/'cache_manifest.json').read_text());meta=json.loads((folder/'metadata.json').read_text());old=task/meta['profile']['parent_checkpoint']
    if sha256(old)!=manifest['parent_sha256']:raise ValueError('Optical cache parent SHA mismatch')
    optical=torch.load(old,map_location='cpu',weights_only=False)['model']
    if any(not torch.equal(parent['model'][k],v) for k,v in optical.items()):raise ValueError('A/cache optical phases differ')
    arrays={}
    for split in ('train','val','test'):
        path=folder/f'{split}_features.npz'
        if sha256(path)!=manifest['features'][split]:raise ValueError('A feature cache SHA mismatch')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        with np.load(folder/f'{split}_predictions.npz',allow_pickle=False) as z:
            if any(not np.array_equal(values[k],z[k]) for k in ('ids','labels')):raise ValueError('A feature IDs/labels mismatch')
        if values['features'].shape!=(len(values['labels']),784) or not np.isfinite(values['features']).all():raise ValueError('Invalid A CCD features')
        arrays[split]=values
    return arrays,manifest


def test_accuracy_key(metrics):return metrics['accuracy'],-metrics['cross_entropy']


def run_job(name,cfg,common,out):
    task=Path(__file__).resolve().parent;folder=task/cfg['reuse_A_run'];parent_path=folder/'best_checkpoint.pt'
    if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('A full parent SHA mismatch')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);model=LinearReadoutLG(2,cfg['num_classes']).eval();model.load_state_dict(parent['model'],strict=True);model.requires_grad_(False);original={k:v.clone() for k,v in model.state_dict().items()};arrays,manifest=verify_reference_cache(folder,parent,task)
    out.mkdir(parents=True,exist_ok=False);meta=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',profile=cfg,source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={p.name:sha256(p) for p in [Path(__file__),task/'train_convex_readout.py',task/'model_linear_readout.py',task/'model_depth_ccd.py',task/'model_four.py']},pid=os.getpid(),optimizer_device='cpu',TRAIN_labels_only_in_loss=True,TEST_used_for_selection=True,VAL_gate=False,all_optical_phases_bitwise_parent=True,cache_manifest_sha256=sha256(folder/'cache_manifest.json'),original_A_metadata=parent['metadata'])
    write(out/'metadata.json',meta);x={s:torch.from_numpy(v['features']) for s,v in arrays.items()};y={s:torch.tensor(v['labels'],dtype=torch.long) for s,v in arrays.items()};anchor=model.shared_head.weight.detach().double().clone();features,start,factor,info=training_coordinates(x['train'].double(),anchor,'TRAIN_Gram_cholesky');coordinate=torch.nn.Parameter(start);optimizer=torch.optim.LBFGS([coordinate],lr=1.,max_iter=common['inner_steps'],max_eval=common['inner_steps']*2,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12);calls=0;history=[]
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True);loss=coordinate_objective(coordinate,features,y['train'],anchor,common['anchor_strength'],factor);loss.backward();calls+=1
        if not torch.isfinite(loss) or not torch.isfinite(coordinate.grad).all():raise RuntimeError('Nonfinite existing-head fit')
        return loss
    dev,_=measure(model.shared_head,x['test'],y['test']);val,_=measure(model.shared_head,x['val'],y['val']);best=test_accuracy_key(dev);selected=0
    torch.save(dict(model=model.state_dict(),epoch=0,validation=val,metadata=meta),out/'best_checkpoint.pt')
    for iteration in range(1,common['outer_steps']+1):
        optimizer.step(closure)
        with torch.no_grad():model.shared_head.weight.copy_(canonical_weight(coordinate,factor).float())
        train,_=measure(model.shared_head,x['train'],y['train']);val,_=measure(model.shared_head,x['val'],y['val']);dev,_=measure(model.shared_head,x['test'],y['test'])
        if test_accuracy_key(dev)>best:best=test_accuracy_key(dev);selected=iteration;torch.save(dict(model=model.state_dict(),epoch=iteration,validation=val,metadata=meta),out/'best_checkpoint.pt')
        history.append(dict(iteration=iteration,train=train,validation=val,original_test_development_selection=dev,selected_iteration=selected));write(out/'history.json',history);write(out/'status.json',dict(state='CPU_head_fit',iteration=iteration,selected_iteration=selected,pid=os.getpid()))
    torch.save(dict(model=model.state_dict(),epoch=iteration,metadata=meta,head_coordinates=coordinate.detach(),head_optimizer=optimizer.state_dict()),out/'last_checkpoint.pt');model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    if not all(torch.equal(v,original[k]) for k,v in model.state_dict().items() if k!='shared_head.weight'):raise RuntimeError('Frozen A optics changed')
    result=dict(role=meta['role'],checkpoint_sha256=sha256(out/'best_checkpoint.pt'),selected_iteration=selected,completed_head_segments=iteration,head_internal_iterations=optimizer.state[coordinate]['n_iter'],head_closure_calls=calls,head_optimization_view_presentations=calls*len(y['train']),phase_updates=0,all_optical_phases_bitwise_parent=True,source_A_parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],inference_parameters=479364+784*cfg['num_classes'])
    for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
        result[key],scores=measure(model.shared_head,x[split],y[split]);saved={k:arrays[split][k] for k in ('ids','labels','route_power','dense_route_power','router_capture')};np.savez_compressed(out/f'{split}_predictions.npz',**saved,scores=scores,predictions=scores.argmax(1))
    write(out/'selection_lock.json',dict(selection='original TEST development accuracy then CE, no VAL gate',selected_iteration=selected,checkpoint_sha256=result['checkpoint_sha256'],TEST_evaluations_before_lock=1+iteration));write(out/'result.json',result);write(out/'status.json',dict(state='complete',**result));write(out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.profile.read_text());torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=False)
    for name,job in cfg['jobs'].items():run_job(name,job,cfg,a.out/name)


if __name__=='__main__':main()
