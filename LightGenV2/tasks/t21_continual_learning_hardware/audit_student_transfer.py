"""Read-only capacity/coverage/numerical diagnosis of sealed B/C distillation.

TRAIN-only least-squares projections diagnose one loss, not a universal capacity
bound. EVAL is queried for disclosed development analysis, never projection fit.
No SDK, GPU, physical update, candidate deployment or source-budget expansion.
"""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .fit_augmented_residual_teacher import residual_teacher_from_artifact
from .project_stage_teacher import held_TRAIN
from .hardware_bridge import load_checkpoint
from .train_native import sha,put


def linear_projection(z,target,tasks):
    """Exact task-balanced LS minimizer; train coordinates are not deployed."""
    tasks=np.asarray(tasks);weights=torch.tensor([1/(len(set(tasks))*int((tasks==n).sum())) for n in tasks],dtype=torch.float64).sqrt()[:,None]
    solution=torch.linalg.lstsq(z.double()*weights,target.double()*weights,driver='gelsd')
    return solution.solution,dict(rank=int(solution.rank),dimensions=z.shape[1],smallest_singular=float(solution.singular_values[-1]),largest_singular=float(solution.singular_values[0]))


def comparison(logits,target,labels):
    lp=logits.argmax(1);tp=target.argmax(1);centered=target-target.mean(1,keepdim=True);pred=logits-logits.mean(1,keepdim=True)
    top=target.topk(2,1).values;prob=F.softmax(target,1);denom=(centered-centered.mean(0,keepdim=True)).square().sum().clamp_min(1e-12)
    return dict(n=len(labels),teacher_correct=int((tp==labels).sum()),student_correct=int((lp==labels).sum()),agreement=float((lp==tp).double().mean()),
        teacher_centered_logit_R2=float(1-(pred-centered).square().sum()/denom),centered_logit_RMSE=float((pred-centered).square().mean().sqrt()),
        teacher_KL_T2=float(F.kl_div(F.log_softmax(logits/2,1),F.softmax(target/2,1),reduction='batchmean')*4),
        teacher_top2_margin_median=float((top[:,0]-top[:,1]).median()),teacher_confidence_mean=float(prob.max(1).values.mean()),
        teacher_entropy_mean=float(-(prob*prob.clamp_min(1e-20).log()).sum(1).mean()))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','teacher_run','student','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--student-sha256',required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());pr=json.loads((a.teacher_run/'protocol.json').read_text());r=json.loads((a.teacher_run/'report.json').read_text());path=a.teacher_run/'offline_teacher.pt'
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or pr['capture_features_SHA256']!=cr['features_sha256']:raise ValueError('Full sealed capture required')
    if not r['teacher_quality_gate_passed'] or r['valid_for_formal_test_table'] is not False or sha(path)!=r['teacher_sha256']:raise ValueError('Disclosed qualified teacher required')
    model,cp=load_checkpoint(a.student,a.student_sha256)
    if cp['architecture']!='d2nn' or cp['measured_features_sha256']!=cr['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured student required')
    artifact=torch.load(path,map_location='cpu',weights_only=False);teacher=residual_teacher_from_artifact(artifact);transform=artifact['transform']
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');selected=pr['selection_source_ids'];exposed=np.array([i for i in te if int(ids[i]) in selected[str(tasks[i])]],dtype=int);unex=np.setdiff1d(te,exposed)
    if len(tr)!=1000+300*(cp['stage']-1) or len(te)!=1000*cp['stage']:raise ValueError('Fixed full phase required')
    z=x@transform.T;target=teacher(z).double();target_centered=target-target.mean(1,keepdim=True);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    held=held_TRAIN(ids[tr],y[tr].numpy(),tasks[tr]);fit=np.setdiff1d(np.arange(len(tr)),held)
    w,rank=linear_projection(z[tr],target_centered[tr],tasks[tr]);held_w,held_rank=linear_projection(z[tr[fit]],target_centered[tr[fit]],tasks[tr[fit]])
    full=z.double()@w;held_pred=z.double()@held_w;folded=(w.T@transform.double()).float();folded_logits=F.linear(x,folded).double();actual=model.shared_head(x).double()
    groups={'TRAIN':tr,'held_TRAIN':tr[held],'all_development':te,'teacher_exposed_development':exposed,'not_in_teacher_gradient_development':unex}
    result={}
    for label,ix in groups.items():
        result[label]={}
        for n in sorted(set(tasks[ix])):
            indices=ix[tasks[ix]==n]
            result[label][n]=dict(actual_student=comparison(actual[indices],target[indices],y[indices]),
                full_TRAIN_linear_projection=comparison(full[indices],target[indices],y[indices]),
                held_TRAIN_linear_projection=comparison(held_pred[indices],target[indices],y[indices]),
                float32_folded_projection=comparison(folded_logits[indices],target[indices],y[indices]),
                projection_fold_argmax_changes=int((full[indices].argmax(1)!=folded_logits[indices].argmax(1)).sum()),
                projection_fold_max_logit_error=float((full[indices]-folded_logits[indices]).abs().max()),
                teacher_coordinate_norm_median=float(z[indices,:-1].double().norm(dim=1).median()),
                actual_student_weight_norm=float(model.shared_head.weight.double().norm()))
    # Query only weak derivatives of original TRAIN for local teacher stability.
    std=x[tr].std(0,unbiased=False);q=(x[tr]+torch.randn_like(x[tr])*std*.001).clamp_min(0);q=q/q.mean(1,keepdim=True)
    teacher_aug=teacher(q@transform.T).double();aug={}
    for n in sorted(set(tasks[tr])):
        ix=np.flatnonzero(tasks[tr]==n);aug[n]=comparison(teacher_aug[ix],target[tr[ix]],y[tr[ix]])
    put(a.out/'protocol.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        capture_features_sha256=cr['features_sha256'],teacher_sha256=r['teacher_sha256'],student_sha256=a.student_sha256,
        original_student_parameters=7840,new_inference_layers=0,projection_fit_uses='Original TRAIN only; held projection excludes held TRAIN',
        selection_source_ids=selected,SDK=False,GPU=False,CPU_threads=2,EVAL_used_for_diagnostic_forward_only=True,
        independent_test_samples=0,valid_for_formal_test_table=False,independent_generalization_claim=False))
    np.savez(a.out/'diagnostic_projection.npz',TRAIN_only_coordinate_weights=w.numpy(),folded_original_weights=folded.numpy())
    put(a.out/'report.json',dict(passed=True,diagnostic_not_deployment=True,full_TRAIN_rank=rank,held_TRAIN_fit_rank=held_rank,
        metrics=result,weak_TRAIN_teacher_stability=aug,LS_is_not_universal_classification_capacity_upper_bound=True,seconds=time.monotonic()-started,
        SDK=False,GPU=False,independent_test_samples=0,valid_for_formal_test_table=False))


if __name__=='__main__':main()
