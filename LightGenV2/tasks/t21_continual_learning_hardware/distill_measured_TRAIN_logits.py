"""TRAIN-only centered-logit regression folded into the existing A readout.

One candidate, five strengths selected inside TRAIN, no TEST teacher or gradient.
The original shared Linear still has7840 parameters; no inference transform.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .hardware_bridge import load_checkpoint
from .distill_measured_TRAIN import teacher_order,validate_teacher
from .refine_measured_ce import contrast_transform,internal_indices
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import sha,put


def ridge_logits(z,target,strength):
    if strength<=0 or not torch.isfinite(z).all() or not torch.isfinite(target).all():
        raise ValueError('Finite TRAIN matrices and positive ridge required')
    centered=target-target.mean(1,keepdim=True)
    penalty=torch.ones(z.shape[1],dtype=z.dtype);penalty[-1]=0
    lhs=z.T@z/len(z)+strength*torch.diag(penalty)
    weight=torch.linalg.solve(lhs,z.T@centered/len(z)).T
    if not torch.isfinite(weight).all():raise RuntimeError('Nonfinite TRAIN ridge solution')
    return weight


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','teacher','teacher_protocol','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args();torch.set_num_threads(2)
    r=json.loads((a.capture/'report.json').read_text());m=json.loads(a.teacher_protocol.read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:
        raise ValueError('Complete immutable measured capture required')
    if sha(a.teacher)!=m['teacher_cache_SHA256']:raise ValueError('Teacher cache changed')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['stage']!=1 or cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:
        raise ValueError('Original A measured readout parent required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        splits=d['splits'].copy();ids=d['source_indices'].copy();tasks=d['tasks'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if x.shape!=(2000,784) or len(tr)!=1000 or len(te)!=1000 or set(tasks)!={'eurosat'} or float((x.mean(1)-1).abs().max())>1e-5:
        raise ValueError('Original mean-one fixed TRAIN1000/TEST1000 only')
    with np.load(a.teacher,allow_pickle=False) as d:
        order=teacher_order(ids[tr],d['source_indices'])
        validate_teacher(m,r['checkpoint_sha256'],ids[tr],y[tr].numpy(),d['source_indices'][order],d['labels'][order])
        target=torch.tensor(d['logits'][order],dtype=torch.float64)
    receipt=json.loads((a.capture/r['stages'][0]/f'{int(tr[0]):05d}'/'receipt.json').read_text())
    if receipt['manifest_sha256']!=m['input_manifest_identity']:raise ValueError('TRAIN manifest changed')
    if target.shape!=(1000,10):raise ValueError('Ten-class source teacher required')
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();f,h=internal_indices(y[tr]);rows=[]
    transform=contrast_transform(x[tr][f]);z=x[tr]@transform.T
    for strength in (.001,.01,.1,1.,10.):
        w=ridge_logits(z[f],target[f],strength);folded=(w@transform).float()
        correct=int(((x[tr][h].float()@folded.T).argmax(1)==y[tr][h]).sum())
        rows.append(dict(strength=strength,TRAIN_holdout_correct=correct,TRAIN_holdout_records=len(h)))
        put(a.out/'status.json',dict(state='TRAIN_internal_centered_logit_fit',latest=rows[-1],updated=time.time()))
    strength=max(rows,key=lambda row:(row['TRAIN_holdout_correct'],row['strength']))['strength']
    transform=contrast_transform(x[tr]);folded=(ridge_logits(x[tr]@transform.T,target,strength)@transform).float()
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,features_SHA256=r['features_sha256'],
        teacher_cache_SHA256=sha(a.teacher),teacher_protocol_SHA256=sha(a.teacher_protocol),teacher_metadata=m,
        method='Class-centered teacher logit MSE, TRAIN-only ridge, exact fold into original Linear',
        TRAIN_internal_selection=rows,selected_strength=strength,TRAIN_records=1000,EVAL_records=1000,
        TRAIN_fit_source_indices=ids[tr][f].tolist(),TRAIN_holdout_source_indices=ids[tr][h].tolist(),
        teacher_TEST_inputs=False,student_TEST_gradients=False,original_graph=True,trainable_parameters=7840,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),
        CPU_threads=2,SDK=False,GPU=False,TEST_selected_development_metric=True,independent_generalization_claim=False))
    with torch.no_grad():
        parent_correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
        model.shared_head.weight.copy_(folded)
        candidate_correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
        train_correct=int((model.shared_head(x[tr].float()).argmax(1)==y[tr]).sum())
    def save(role):
        if upstream_state(model)!=before:raise RuntimeError('Frozen optical upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='TRAIN-only centered teacher logit ridge',teacher_TRAIN_cache_SHA256=sha(a.teacher),
            teacher_seen_TEST_inputs=False,TEST_gradients=False,TEST_selected_development_metric=True)
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
    save('last')
    if candidate_correct<=parent_correct:model.load_state_dict(cp['model'],strict=True)
    save('best');scores={}
    for role,expected in (('best',max(parent_correct,candidate_correct)),('last',candidate_correct)):
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved upstream identity changed')
        value=evaluate(loaded.shared_head,x.float(),y,tasks,te)
        if value['eurosat']['correct']!=expected:raise RuntimeError('Saved CPU original-graph replay differs')
        put(a.out/(role+'_test.json'),dict(scores=value,source_indices=ids[te].tolist(),TEST_selected_development_metric=True));scores[role]=brief(value)
    result=dict(passed=True,parent_correct=parent_correct,candidate_correct=candidate_correct,best_correct=max(parent_correct,candidate_correct),
        candidate_TRAIN_correct=train_correct,best=scores['best'],last=scores['last'],retained_parent=candidate_correct<=parent_correct,
        strict_CPU_reloaded=True,frozen_upstream=True,original_graph=True,trainable_parameters=7840,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),selected_strength=strength,
        teacher_TEST_inputs=False,TEST_gradients=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
        seconds=time.time()-started,SDK=False,GPU=False)
    put(a.out/'report.json',result);put(a.out/'status.json',dict(state='complete',updated=time.time()));print(json.dumps(result))


if __name__=='__main__':main()
