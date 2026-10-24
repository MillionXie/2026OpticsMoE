"""Frozen S1 TRAIN soft targets -> original measured7840-parameter A readout.

No TEST teacher, augmentation derived from TEST, new student layer, SDK or GPU.
Four fixed strengths are selected inside TRAIN; only then is the one resulting
candidate compared to its parent on the full development TEST. Negative results
retain the parent. The invertible conditioning folds into the original Linear.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .hardware_bridge import load_checkpoint
from .refine_measured_ce import contrast_transform,internal_indices
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import sha,put


def validate_teacher(metadata,source_checkpoint,ids,labels,teacher_ids,teacher_labels):
    if not metadata.get('passed') or metadata.get('TEST_used') is not False or metadata.get('TEST_augmented') is not False:
        raise ValueError('Only source-S1 TRAIN-only teacher information admitted')
    if metadata['checkpoint_SHA256']!=source_checkpoint or metadata['TRAIN_records']!=1000:
        raise ValueError('Teacher checkpoint or TRAIN identity differs')
    if not np.array_equal(ids,teacher_ids) or not np.array_equal(labels,teacher_labels):
        raise ValueError('Teacher and physical TRAIN record order differs')


def teacher_order(physical_ids,teacher_ids):
    physical_ids=np.asarray(physical_ids);teacher_ids=np.asarray(teacher_ids)
    if (len(np.unique(physical_ids))!=len(physical_ids) or len(np.unique(teacher_ids))!=len(teacher_ids) or
        len(physical_ids)!=len(teacher_ids) or set(physical_ids.tolist())!=set(teacher_ids.tolist())):
        raise ValueError('TRAIN source IDs are not the same unique set')
    lookup={int(value):i for i,value in enumerate(teacher_ids)}
    return np.array([lookup[int(value)] for value in physical_ids],dtype=np.int64)


def objective(logits,labels,teacher):
    ce=F.cross_entropy(logits,labels,label_smoothing=.02)
    kd=F.kl_div(F.log_softmax(logits/2.,1),F.softmax(teacher.detach()/2.,1),reduction='batchmean')*4.
    return .2*ce+4.*kd


def noise_penalty(weight,noise):
    if noise is None:return weight.new_zeros(())
    perturbation=noise@weight.T
    perturbation=perturbation-perturbation.mean(1,keepdim=True)
    return perturbation.square().mean()


def fit(z,y,teacher,strength,steps,noise=None):
    weight=torch.zeros(teacher.shape[1],z.shape[1],dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([weight],lr=1.,max_iter=steps,max_eval=steps*2,
        history_size=20,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    started=time.monotonic()
    def closure():
        if time.monotonic()-started>90:raise RuntimeError('Finite TRAIN optimization bound reached')
        optimizer.zero_grad(set_to_none=True)
        loss=objective(z@weight.T,y,teacher)+.5*strength*weight[:,:-1].square().sum()/teacher.shape[1]
        loss=loss+noise_penalty(weight,noise)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN KD objective')
        loss.backward();return loss
    optimizer.step(closure)
    return weight.detach(),float(closure().detach())


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','teacher','teacher_protocol','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True)
    p.add_argument('--noise',type=Path);p.add_argument('--noise-protocol',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if bool(a.noise)!=bool(a.noise_protocol):raise ValueError('Noise array and provenance both required')
    report=json.loads((a.capture/'report.json').read_text());metadata=json.loads(a.teacher_protocol.read_text())
    if not report['passed'] or report['pilot'] or not report['SDK_released'] or sha(a.capture/'features.npz')!=report['features_sha256']:
        raise ValueError('Complete immutable physical capture required')
    if sha(a.parent)!=a.parent_sha256 or sha(a.teacher)!=metadata['teacher_cache_SHA256']:
        raise ValueError('Parent or TRAIN teacher cache SHA differs')
    model,checkpoint=load_checkpoint(a.parent,a.parent_sha256)
    if checkpoint['stage']!=1 or checkpoint.get('measured_features_sha256')!=report['features_sha256']:
        raise ValueError('A-only original measured parent required')
    if model.shared_head.weight.numel()!=7840:raise ValueError('Changed original readout capacity')
    before=upstream_state(model)
    with np.load(a.capture/'features.npz',allow_pickle=False) as data:
        x=torch.tensor(data['features'],dtype=torch.float64);y=torch.tensor(data['labels'],dtype=torch.long)
        splits=data['splits'].copy();ids=data['source_indices'].copy();tasks=data['tasks'].copy()
    train=np.flatnonzero(splits=='train');test=np.flatnonzero(splits=='test')
    if len(train)!=1000 or len(test)!=1000 or set(tasks)!={'eurosat'} or x.shape!=(2000,784):
        raise ValueError('Fixed A TRAIN1000 / TEST1000 only')
    with np.load(a.teacher,allow_pickle=False) as data:
        order=teacher_order(ids[train],data['source_indices'])
        validate_teacher(metadata,report['checkpoint_sha256'],ids[train],y[train].numpy(),data['source_indices'][order],data['labels'][order])
        target=torch.tensor(data['logits'][order],dtype=torch.float64)
    if target.shape!=(1000,10) or not torch.isfinite(target).all():raise ValueError('Invalid TRAIN teacher logits')
    noise=None;noise_metadata=None
    if a.noise:
        noise_metadata=json.loads(a.noise_protocol.read_text())
        cp=json.loads((a.capture/'protocol.json').read_text())
        if (noise_metadata.get('TEST_used') is not False or noise_metadata.get('TRAIN_only') is not True or
            noise_metadata['capture_features_SHA256']!=report['features_sha256'] or
            noise_metadata['hardware_SHA256']!=cp['hardware_sha256'] or
            noise_metadata['source_checkpoint_SHA256']!=report['checkpoint_sha256'] or
            noise_metadata['noise_array_SHA256']!=sha(a.noise)):
            raise ValueError('Measured TRAIN noise identity differs')
        with np.load(a.noise,allow_pickle=False) as data:noise=torch.tensor(data['deltas'],dtype=torch.float64)
        if noise.shape!=(16,784) or not torch.isfinite(noise).all() or float(noise.mean(1).abs().max())>1e-5:
            raise ValueError('Fixed four TRAIN payloads times four repeats, original mean-one coordinates required')
    if float((x.mean(1)-1).abs().max())>1e-5:raise ValueError('Original mean-one features required')
    source=json.loads((a.capture/report['stages'][0]/f'{int(train[0]):05d}'/'receipt.json').read_text())
    if source['manifest_sha256']!=metadata['input_manifest_identity']:raise ValueError('TRAIN manifest identity differs')
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();f,h=internal_indices(y[train])
    transform=contrast_transform(x[train][f]);z=x[train]@transform.T;rows=[]
    for strength in (.001,.01,.1,1.):
        w,loss=fit(z[f],y[train][f],target[f],strength,80,None if noise is None else noise@transform.T)
        folded=(w@transform).float()
        correct=int(((x[train][h].float()@folded.T).argmax(1)==y[train][h]).sum())
        rows.append(dict(strength=strength,TRAIN_holdout_correct=correct,TRAIN_holdout_records=len(h),TRAIN_loss=loss))
        put(a.out/'status.json',dict(state='TRAIN_internal_KD',latest=rows[-1],updated=time.time()))
    strength=max(rows,key=lambda r:(r['TRAIN_holdout_correct'],r['strength']))['strength']
    transform=contrast_transform(x[train]);w,loss=fit(x[train]@transform.T,y[train],target,strength,120,None if noise is None else noise@transform.T)
    folded=(w@transform).float()
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,features_SHA256=report['features_sha256'],
        teacher_cache_SHA256=sha(a.teacher),teacher_protocol_SHA256=sha(a.teacher_protocol),teacher_metadata=metadata,
        teacher_information='Corresponding frozen native S1 on TRAIN1000 only; original S1 checkpoint TEST-selected development',
        teacher_TEST_inputs=False,teacher_TEST_augmentation=False,student_TEST_gradients=False,
        strengths=(.001,.01,.1,1.),selected_strength=strength,TRAIN_internal_selection=rows,
        TRAIN_fit_source_indices=ids[train][f].tolist(),TRAIN_holdout_source_indices=ids[train][h].tolist(),
        loss='0.2 smoothed CE +4 KL(T=2), fixed before TEST',TRAIN_records=1000,EVAL_records=1000,
        measured_TRAIN_noise_metadata=noise_metadata,measured_noise_logit_variance_weight=1. if noise is not None else 0.,
        noise_definition='TRAI-only within-payload residuals; class-common logit component removed, no added inference preprocessing',
        original_graph=True,trainable_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,CPU_threads=2,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),
        TEST_selected_development_metric=True,independent_generalization_claim=False))
    with torch.no_grad():
        parent_correct=int((model.shared_head(x[test].float()).argmax(1)==y[test]).sum())
        model.shared_head.weight.copy_(folded)
        candidate_correct=int((model.shared_head(x[test].float()).argmax(1)==y[test]).sum())
        train_correct=int((model.shared_head(x[train].float()).argmax(1)==y[train]).sum())
    def save(name):
        if upstream_state(model)!=before:raise RuntimeError('Optical/upstream state changed')
        saved=copy.deepcopy(checkpoint);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='TRAIN-only source S1 distillation folded into original Linear',
            teacher_TRAIN_cache_SHA256=sha(a.teacher),teacher_seen_TEST_inputs=False,TEST_gradients=False,
            measured_TRAIN_noise_SHA256=sha(a.noise) if a.noise else None,
            TEST_selected_development_metric=True)
        torch.save(saved,a.out/(name+'_checkpoint.pt'))
    save('last')
    if candidate_correct<=parent_correct:model.load_state_dict(checkpoint['model'],strict=True)
    save('best')
    scores={}
    for name,expected in (('best',max(parent_correct,candidate_correct)),('last',candidate_correct)):
        saved,_=load_checkpoint(a.out/(name+'_checkpoint.pt'),sha(a.out/(name+'_checkpoint.pt')))
        if upstream_state(saved)!=before:raise RuntimeError('Saved frozen-state mismatch')
        value=evaluate(saved.shared_head,x.float(),y,tasks,test)
        if value['eurosat']['correct']!=expected:raise RuntimeError('Strict CPU original-graph replay differs')
        put(a.out/(name+'_test.json'),dict(scores=value,source_indices=ids[test].tolist(),TEST_selected_development_metric=True))
        scores[name]=brief(value)
    result=dict(passed=True,parent_correct=parent_correct,candidate_correct=candidate_correct,
        candidate_TRAIN_correct=train_correct,best_correct=max(parent_correct,candidate_correct),best=scores['best'],last=scores['last'],
        retained_parent=candidate_correct<=parent_correct,strict_CPU_reloaded=True,frozen_upstream=True,original_graph=True,
        trainable_parameters=7840,teacher_TEST_inputs=False,TEST_gradients=False,TEST_selected_development_metric=True,
        measured_TRAIN_noise=noise is not None,
        independent_generalization_claim=False,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),
        selected_strength=strength,TRAIN_loss=loss,seconds=time.time()-started,GPU=False,SDK=False)
    put(a.out/'report.json',result);put(a.out/'status.json',dict(state='complete',updated=time.time()));print(json.dumps(result))


if __name__=='__main__':main()
