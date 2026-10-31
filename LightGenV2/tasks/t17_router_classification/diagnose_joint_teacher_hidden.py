"""CPU-only original-TRAIN hidden alignment; no student or teacher updates."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from .train_original_test_teacher import teacher_head
from .semantic_distillation import ridge_projection
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def linear_CKA(x,y):
    x=x.double()-x.double().mean(0);y=y.double()-y.double().mean(0)
    return float((x.T@y).square().sum()/((x.T@x).norm()*(y.T@y).norm()).clamp_min(1e-30))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-run',required=True,type=Path);parser.add_argument('--teacher-run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();torch.set_num_threads(4)
    parent=args.parent_run;teacher=args.teacher_run;task=Path(__file__).resolve().parent
    result=json.loads((parent/'result.json').read_text());ck=torch.load(parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if sha256(parent/'best_checkpoint.pt')!=result['checkpoint_sha256']:raise ValueError('Student PT SHA')
    manifest=json.loads((parent/'cache_manifest.json').read_text());cfg=ck['metadata']['head_refit_profile'];cached_parent=task/cfg['parent_checkpoint']
    if sha256(cached_parent)!=manifest['parent_checkpoint_sha256']:raise ValueError('CCD cache parent SHA')
    original=torch.load(cached_parent,map_location='cpu',weights_only=False)
    if any(not torch.equal(v,ck['model'][name]) for name,v in original['model'].items() if name!='shared_head.weight'):raise ValueError('CCD cache phase mismatch')
    for name in ['model_four.py','model_depth_ccd.py','model_linear_readout.py']:
        if manifest['source_sha256'][name]!=sha256(task/name):raise ValueError('CCD optical source SHA')
    if sha256(parent/'train_features.npz')!=manifest['feature_sha256']['train']:raise ValueError('CCD feature SHA')
    values=np.load(parent/'train_features.npz');x=torch.from_numpy(values['ccd_features']).float()
    tm=json.loads((teacher/'cache_manifest.json').read_text())
    for name in ['best_checkpoint.pt','features_train.npy','identities_train.npz','teacher_logits_train.npz']:
        if sha256(teacher/name)!=tm[name]:raise ValueError('Teacher source SHA: '+name)
    ids=np.load(teacher/'identities_train.npz')
    if any(not np.array_equal(values[k],ids[k]) for k in ['ids','labels','domains']):raise ValueError('Joint source TRAIN identity')
    saved=torch.load(teacher/'best_checkpoint.pt',map_location='cpu',weights_only=False);head=teacher_head(10).eval();head.load_state_dict(saved['head'],strict=True);head.requires_grad_(False)
    f=np.load(teacher/'features_train.npy',mmap_mode='r');hidden=[]
    with torch.no_grad():
        flat=f.reshape(-1,512)
        for start in range(0,len(flat),1024):hidden.append(head[:-1](torch.from_numpy(flat[start:start+1024].copy()).float()))
        hidden=torch.cat(hidden).reshape(len(x),8,256)
        logits=head[-1](hidden[:,0]);target=np.load(teacher/'teacher_logits_train.npz')['logits'][:,0]
        if not np.array_equal(logits.argmax(1).numpy(),target.argmax(1)):raise ValueError('Shared teacher hidden replay')
    mean=hidden.mean((0,1));std=hidden.std((0,1)).clamp_min(.001);y=((hidden[:,0]-mean)/std).float()
    fit=[];hold=[];rng=np.random.default_rng(17)
    for d in (0,1):
        for c in range(10):
            subset=np.flatnonzero((values['domains']==d)&(values['labels']==c));rng.shuffle(subset);boundary=len(subset)//5;hold.extend(subset[:boundary]);fit.extend(subset[boundary:])
    weight,info=ridge_projection(x[fit],y[fit]);prediction=x[hold]@weight.T;rows={}
    for d,name in [(0,'RGB'),(1,'SAR')]:
        all_mask=values['domains']==d;mask=all_mask[np.asarray(hold)];truth=y[hold][mask];estimate=prediction[mask]
        rows[name]=dict(original_TRAIN_images=int(all_mask.sum()),heldout_TRAIN_images=int(mask.sum()),CCD_to_shared_GELU256_linear_CKA=linear_CKA(x[all_mask],y[all_mask]),heldout_TRAIN_hidden_R2=float(1-(estimate-truth).square().sum()/(truth-truth.mean(0)).square().sum().clamp_min(1e-20)),hidden_MSE=float((estimate-truth).square().mean()))
    initialization,full_info=ridge_projection(x,y)
    args.out.mkdir(parents=True,exist_ok=False);torch.save(dict(training_only_projection_weight=initialization,teacher_TRAIN8_mean=mean,teacher_TRAIN8_std=std),args.out/'training_only_projection_initialization.pt')
    report=dict(role='ORIGINAL_TRAIN_ONLY_READONLY_DIAGNOSIS',source_commit=os.environ.get('T17_SOURCE_COMMIT'),pid=os.getpid(),device='CPU only; no CUDA context',parent_checkpoint_sha256=result['checkpoint_sha256'],teacher_checkpoint_sha256=tm['best_checkpoint.pt'],CCD_features_sha256=sha256(parent/'train_features.npz'),teacher_features_sha256=tm['features_train.npy'],TRAIN_images=len(x),TRAIN_regression_fit_images=len(fit),TRAIN_regression_holdout_images=len(hold),VAL_TEST_statistics=0,student_updates=0,teacher_updates=0,all_optical_phases_match_verified_CCD_cache=True,teacher_hidden_before_final_Linear='GELU256; frozen teacher eval; one jointly trained head',teacher_TRAIN_head_predictions_exact=True,teacher_TRAIN_logits_max_abs=float(np.abs(logits.numpy()-target).max()),normalization_parts='original TRAIN all8 views only',ridge_fit=info,full_TRAIN_initialization=full_info,extra_projection_training_parameters=784*256,projection_in_student_inference=False,domains=rows,interpretation='TRAIN feature reconstruction diagnostic, not classification accuracy, new TEST results, convergence or a capacity bound. A future training-only feature auxiliary is a separate finite hypothesis, not an active run.')
    write(args.out/'diagnosis.json',report);write(args.out/'manifest.json',dict(files={p.name:sha256(p) for p in args.out.iterdir() if p.is_file()},parent_checkpoint_sha256=result['checkpoint_sha256']));print(json.dumps(report))


if __name__=='__main__':main()
