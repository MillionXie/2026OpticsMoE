"""Bounded canonical-view refit of the existing Linear; TEST-development selects.

Optics stay frozen. TRAIN contributes hard CE; optional original TEST contributes
detached teacher KL only. The numerical coordinates are folded back into W.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .alternating_readout import anchored_classification
from .train_convex_readout import cache, training_coordinates, canonical_weight
from .train_figure_revision import build_model
from .readout_data import load_raw
from .figure_revision_data import load_panel_f
from .train_linear_readout import measure
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def selection_key(metrics, joint):
    if joint:
        return min(v['accuracy'] for v in metrics['domains'].values()), metrics['accuracy'], -metrics['cross_entropy']
    return metrics['accuracy'], -metrics['cross_entropy']


def fit_objective(weight, features, labels, anchor, teacher=None, hard_mask=None):
    return anchored_classification(F.linear(features, weight), labels, weight, anchor,
        .001, 0., teacher, .5 if teacher is not None else 0., 2., hard_mask)


def initialization_profile(parent,cfg):
    profile=parent['metadata']['profile'];requested=cfg.get('initialize_top_k')
    if requested is None:
        if cfg.get('changed_k_initialization'):raise ValueError('Changed-k flag requires an explicit actual k')
        return profile
    previous=profile.get('top_k',2)
    if type(requested) is not int or not cfg.get('changed_k_initialization') or profile.get('dataset')!='kather2016' or profile.get('architecture')!='LG' or profile.get('main_layers')!=2 or (previous,requested) not in ((1,2),(3,4)):
        raise ValueError('Explicit adjacent Kather k1->2 or k3->4 optical initialization required')
    return dict(profile,top_k=requested)


def run_job(cfg, out):
    task=Path(__file__).resolve().parent
    parent_path=task/cfg['parent_checkpoint']
    if sha256(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Parent SHA mismatch')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    profile=initialization_profile(parent,cfg)
    if profile['architecture']!='LG' or profile['main_layers']!=2:raise ValueError('Two-layer original LG required')
    joint=profile['dataset']=='eurosat_original_f'
    model=build_model(profile).cuda().eval();model.load_state_dict(parent['model'],strict=True);model.requires_grad_(False)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    out.mkdir(parents=True,exist_ok=False)
    raw,identity=load_panel_f(profile['data']) if joint else load_raw(Path(profile['data_root']),profile)
    meta=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',profile=profile,head_refit_profile=cfg,
        source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={n:sha256(task/n) for n in
        [Path(__file__).name,'model_four.py','model_linear_readout.py','model_depth_ccd.py','alternating_readout.py']},
        pid=os.getpid(),data=identity,phase_updates=0,TEST_used_for_selection=True,VAL_gate=False,
        original_TEST_soft_in_loss=joint,TEST_hard_GT_in_loss=False,canonical_code0_only=True,
        maximum_internal_iterations=150,optimization_anchor='initial existing head; .001/2 Frobenius square')
    meta['changed_k_initialization']=bool(cfg.get('changed_k_initialization'))
    meta['parent_inference_top_k']=parent['metadata']['profile'].get('top_k',2)
    meta['actual_inference_top_k']=profile.get('top_k',2)
    write(out/'metadata.json',meta)
    arrays={s:cache(model,part,out/f'{s}_features.npz',64) for s,part in raw.items()}
    write(out/'cache_manifest.json',dict(parent_checkpoint_sha256=sha256(parent_path),
        source_sha256=meta['source_sha256'],feature_sha256={s:sha256(out/f'{s}_features.npz') for s in arrays}))
    x={s:torch.from_numpy(a['ccd_features']).cuda() for s,a in arrays.items()}
    y={s:torch.tensor(a['labels'],device='cuda',dtype=torch.long) for s,a in arrays.items()}
    def metrics(split):
        result,scores=measure(model.shared_head,x[split],y[split])
        if joint:
            domains=arrays[split]['domains'];pred=scores.argmax(1)
            result['domains']={name:dict(n=int((domains==d).sum()),correct=int((pred[domains==d]==arrays[split]['labels'][domains==d]).sum()),accuracy=float((pred[domains==d]==arrays[split]['labels'][domains==d]).mean())) for d,name in [(0,'RGB'),(1,'SAR')]}
        return result,scores
    # This exact cache/float32 head must reproduce the original complete predictions.
    initial,_=metrics('test');expected=parent_path.parent/'test_predictions.npz'
    z=np.load(expected);_,score=metrics('test')
    if any(not np.array_equal(arrays['test'][name],z[name]) for name in ('ids','labels')):raise ValueError('Complete fixed parent TEST IDs/labels differ')
    if not cfg.get('changed_k_initialization') and not np.array_equal(score.argmax(1),z['predictions']):raise RuntimeError('Canonical cache parent replay mismatch')
    meta['initial_current_k_TEST_development']=initial
    meta['parent_prediction_replay_required']=not bool(cfg.get('changed_k_initialization'))
    teacher=None;mask=None;train_x=x['train'].double();labels=y['train']
    if joint:
        targets=[]
        for split,filename in [('train','TRAIN_teacher_logits.npz'),('test','SOURCE_TEST_teacher_logits.npz')]:
            path=parent_path.parent/filename
            manifest_path=parent_path.parent/('teacher_manifest.json' if split=='train' else 'source_TEST_distillation_manifest.json')
            manifest=json.loads(manifest_path.read_text())
            if sha256(path)!=manifest['logits_sha256' if split=='train' else 'cache_sha256']:raise ValueError('Teacher logits SHA mismatch')
            values=np.load(path)
            for key in ('ids','labels','domains'):
                if not np.array_equal(values[key],arrays[split][key]):raise ValueError('Teacher cache identity mismatch')
            logits=values['logits'] if 'logits' in values else values['teacher_logits']
            if logits.shape!=(len(y[split]),8,10):raise ValueError('Joint 8-view logits required')
            targets.append(torch.from_numpy(logits[:,0]).cuda().double())
            # Preserve the exact input targets with their SHA independently of model state.
            import shutil
            shutil.copyfile(path,out/filename)
        teacher=torch.cat(targets).detach();train_x=torch.cat([train_x,x['test'].double()])
        labels=torch.cat([y['train'],torch.full_like(y['test'],-1)])
        mask=labels>=0
        meta['teacher_cache_SHA256']={name:sha256(out/name) for name in ['TRAIN_teacher_logits.npz','SOURCE_TEST_teacher_logits.npz']}
    anchor=model.shared_head.weight.detach().double().clone()
    transformed,start,factor,coordinates=training_coordinates(train_x,anchor,'TRAIN_Gram_cholesky')
    # Gram includes soft-input TEST only in the explicitly disclosed joint mode.
    meta['coordinate_statistics']=coordinates;meta['Gram_includes_original_TEST_soft_inputs']=joint
    write(out/'metadata.json',meta)
    coordinate=torch.nn.Parameter(start)
    opt=torch.optim.LBFGS([coordinate],lr=1.,max_iter=5,max_eval=10,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
    calls=0
    def objective(w,features):return fit_objective(w,features,labels,anchor,teacher,mask)
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        weight=canonical_weight(coordinate,factor)
        # Classification in numerical coordinates, regularizer always in original W.
        loss=anchored_classification(F.linear(transformed,coordinate),labels,weight,anchor,.001,0.,teacher,.5 if joint else 0.,2.,mask)
        loss.backward();calls+=1
        if not torch.isfinite(loss) or not torch.isfinite(coordinate.grad).all():raise RuntimeError('Nonfinite fit')
        return loss
    w=anchor.detach().requires_grad_(True);initial_gradient=torch.autograd.grad(objective(w,train_x),w)[0]
    meta['initial_canonical_gradient_max_abs']=float(initial_gradient.abs().max());write(out/'metadata.json',meta)
    best=selection_key(initial,joint);selected=0;history=[]
    def save(path,iteration):torch.save(dict(model=model.state_dict(),epoch=iteration,metadata=meta),path)
    save(out/'best_checkpoint.pt',0)
    for iteration in range(1,31):
        opt.step(closure)
        with torch.no_grad():model.shared_head.weight.copy_(canonical_weight(coordinate,factor).float())
        dev,_=metrics('test');trn,_=metrics('train');val,_=metrics('val')
        if selection_key(dev,joint)>best:best=selection_key(dev,joint);selected=iteration;save(out/'best_checkpoint.pt',iteration)
        row=dict(iteration=iteration,train=trn,validation=val,original_test_development_selection=dev,selected_iteration=selected,inner_iterations=opt.state[coordinate]['n_iter'])
        history.append(row);write(out/'history.json',history);write(out/'status.json',dict(state='bounded_canonical_head_fit',pid=os.getpid(),**row))
    save(out/'last_checkpoint.pt',iteration)
    model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.state_dict().items() if k!='shared_head.weight')
    result=dict(role=meta['role'],checkpoint_sha256=sha256(out/'best_checkpoint.pt'),selected_iteration=selected,
        completed_head_segments=30,head_internal_iterations=opt.state[coordinate]['n_iter'],head_closure_calls=calls,
        phase_updates=0,all_optical_phases_bitwise_parent=True,parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],
        canonical_fit_presentations=calls*len(labels),original_TRAIN_hard_images=len(y['train']),original_TEST_soft_images=len(y['test']) if joint else 0)
    for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
        result[key],scores=metrics(split);saved={n:v for n,v in arrays[split].items() if n!='ccd_features'}
        np.savez_compressed(out/f'{split}_predictions.npz',**saved,scores=scores,predictions=scores.argmax(1))
    write(out/'selection_lock.json',dict(selection='original TEST development only, no VAL gate',selected_iteration=selected,checkpoint_sha256=result['checkpoint_sha256'],TEST_evaluations_before_lock=31))
    write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),**result));write(out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))
    del model,x,y,coordinate,opt,transformed,train_x;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.profile.read_text());require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4)
    for name,job in cfg['jobs'].items():run_job(job,a.out/name)


if __name__=='__main__':main()
