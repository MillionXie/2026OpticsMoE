"""One TRAIN-selected shrinkage-discriminant candidate, original A Linear.

Shared within-class covariance supplies a generative inductive bias rather
than another TEST-driven loss sweep. Bias/conditioning fold into mean-one784.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .hardware_bridge import load_checkpoint
from .refine_measured_ce import contrast_transform,internal_indices
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import sha,put


def discriminant_weight(z,y,shrinkage):
    if not 0<shrinkage<=1 or z.dtype!=torch.float64 or float((z[:,-1]-1).abs().max())>1e-5:
        raise ValueError('Double precision mean-one DC and positive shrinkage required')
    value=z[:,:-1];counts=torch.bincount(y,minlength=10)
    if len(counts)!=10 or not torch.equal(counts,counts[0].expand(10)) or int(counts.min())<2:
        raise ValueError('Balanced ten-class TRAIN only')
    means=torch.stack([value[y==k].mean(0) for k in range(10)])
    residual=value-means[y];cov=residual.T@residual/(len(y)-10)
    scale=cov.diag().mean().clamp_min(1e-12)
    cov=(1-shrinkage)*cov+shrinkage*scale*torch.eye(cov.shape[0],dtype=cov.dtype)
    weights=torch.linalg.solve(cov,means.T).T;bias=-.5*(weights*means).sum(1)
    result=torch.cat((weights,bias[:,None]),1)
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite TRAIN discriminant solution')
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args();torch.set_num_threads(2)
    r=json.loads((a.capture/'report.json').read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:
        raise ValueError('Complete immutable capture required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['stage']!=1 or cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:
        raise ValueError('Original measured A parent required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        tasks=d['tasks'].copy();splits=d['splits'].copy();ids=d['source_indices'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if x.shape!=(2000,784) or len(tr)!=1000 or len(te)!=1000 or set(tasks)!={'eurosat'} or float((x.mean(1)-1).abs().max())>1e-5:
        raise ValueError('Fixed mean-one A TRAIN1000/TEST1000 only')
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();f,h=internal_indices(y[tr]);rows=[]
    transform=contrast_transform(x[tr][f]);z=x[tr]@transform.T
    for strength in (.01,.1,.5,1.):
        folded=(discriminant_weight(z[f],y[tr][f],strength)@transform).float()
        correct=int(((x[tr][h].float()@folded.T).argmax(1)==y[tr][h]).sum())
        rows.append(dict(shrinkage=strength,TRAIN_holdout_correct=correct,TRAIN_holdout_records=len(h)))
        put(a.out/'status.json',dict(state='TRAIN_internal_covariance_selection',latest=rows[-1],updated=time.time()))
    strength=max(rows,key=lambda row:(row['TRAIN_holdout_correct'],row['shrinkage']))['shrinkage']
    transform=contrast_transform(x[tr]);folded=(discriminant_weight(x[tr]@transform.T,y[tr],strength)@transform).float()
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,features_SHA256=r['features_sha256'],
        method='Equal-prior ten-class Gaussian shared-covariance discriminant with TRAIN-only shrinkage and conditioning',
        TRAIN_internal_selection=rows,selected_shrinkage=strength,TRAIN_fit_source_indices=ids[tr][f].tolist(),TRAIN_holdout_source_indices=ids[tr][h].tolist(),
        TRAIN_records=1000,EVAL_records=1000,TEST_gradients=False,teacher_TEST_inputs=False,teacher_used=False,
        original_graph=True,trainable_parameters=7840,new_inference_layers=0,bias_folded_into_original_mean_one_coordinates=True,
        CPU_threads=2,SDK=False,GPU=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),
        TEST_selected_development_metric=True,independent_generalization_claim=False))
    with torch.no_grad():
        parent_correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
        model.shared_head.weight.copy_(folded);candidate_correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
        train_correct=int((model.shared_head(x[tr].float()).argmax(1)==y[tr]).sum())
    def save(role):
        if upstream_state(model)!=before:raise RuntimeError('Frozen upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='TRAIN-only shrinkage shared-covariance discriminant folded into original Linear',TEST_gradients=False,TEST_selected_development_metric=True)
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
    save('last')
    if candidate_correct<=parent_correct:model.load_state_dict(cp['model'],strict=True)
    save('best');scores={}
    for role,expected in (('best',max(parent_correct,candidate_correct)),('last',candidate_correct)):
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved upstream differs')
        value=evaluate(loaded.shared_head,x.float(),y,tasks,te)
        if value['eurosat']['correct']!=expected:raise RuntimeError('Saved original CPU graph differs')
        scores[role]=brief(value);put(a.out/(role+'_test.json'),dict(scores=value,source_indices=ids[te].tolist(),TEST_selected_development_metric=True))
    result=dict(passed=True,parent_correct=parent_correct,candidate_correct=candidate_correct,best_correct=max(parent_correct,candidate_correct),
        candidate_TRAIN_correct=train_correct,best=scores['best'],last=scores['last'],retained_parent=candidate_correct<=parent_correct,
        strict_CPU_reloaded=True,frozen_upstream=True,original_graph=True,trainable_parameters=7840,selected_shrinkage=strength,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_gradients=False,teacher_used=False,
        TEST_selected_development_metric=True,independent_generalization_claim=False,seconds=time.time()-started,SDK=False,GPU=False)
    put(a.out/'report.json',result);put(a.out/'status.json',dict(state='complete',updated=time.time()));print(json.dumps(result))


if __name__=='__main__':main()
