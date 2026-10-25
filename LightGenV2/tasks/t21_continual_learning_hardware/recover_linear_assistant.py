"""Recover a provisional B Linear assistant from TRAIN queries only.

The assistant missed the absolute goal but improved the whole-stage deficit.
This entry does not relax admission: the recovered student remains provisional
unless every original score band passes. Inherited TEST information is explicit.
"""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
import torch
from .audit_student_transfer import linear_projection
from .distill_capacity_matched_stage import TRAIN_queries
from .fit_gated_linear_teacher import DISCLOSURE
from .fit_B_balanced_residual_teacher import whole_stage_selection
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import put,sha


def within_bands(scores):
    reference={'eurosat':.533,'clevr':.760}
    return set(scores)==set(reference) and all(reference[n]-.05-1e-7<=r['accuracy']<reference[n] for n,r in scores.items())


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','assistant_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Inherited TEST information must be disclosed')
    torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());pr=json.loads((a.assistant_run/'protocol.json').read_text());ar=json.loads((a.assistant_run/'report.json').read_text())
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256']:raise ValueError('Sealed capture required')
    expected=json.loads(Path(__file__).with_name('config_capacity_matched_distillation_20261025.json').read_text())
    if pr['mode']!='assistant' or pr['profile']!=expected or pr['parent_SHA256']!=a.parent_sha256 or pr['capture_features_SHA256']!=cr['features_sha256'] or not ar['passed'] or ar['valid_for_formal_test_table'] is not False:raise ValueError('Exact fixed-scope soft assistant required')
    path=a.assistant_run/'assistant_best_checkpoint.pt';assistant,ap=load_checkpoint(path,sha(path));parent,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(parent)
    if cp['architecture']!='d2nn' or cp['stage']!=2 or cp['measured_features_sha256']!=cr['features_sha256'] or upstream_state(assistant)!=before:raise ValueError('Same B measured prefix required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if len(tr)!=1300 or len(te)!=2000 or any(int((tasks[te]==n).sum())!=1000 for n in ('eurosat','clevr')):raise ValueError('Fixed complete B required')
    baseline=evaluate(parent.shared_head,x,y,tasks,te);original=evaluate(assistant.shared_head,x,y,tasks,te);floors={'eurosat':.483,'clevr':.710}
    if whole_stage_selection(original,floors)<=whole_stage_selection(baseline,floors):raise ValueError('Assistant does not improve the fixed whole-stage deficit')
    q,qt=TRAIN_queries(x[tr],tasks[tr],expected)
    # The sealed teacher's conditioning coordinates are not needed: reconstruct
    # in an invertible TRAIN-only coordinate system to avoid F32 LS roundoff.
    from .fit_gated_linear_teacher import full_covariance_transform
    from .refine_measured_ce import contrast_transform
    transform,_=full_covariance_transform(x[tr],contrast_transform(x[tr]));target=q.double()@assistant.shared_head.weight.double().T;target-=target.mean(1,keepdim=True)
    w,rank=linear_projection(q.double()@transform.T,target,qt)
    if rank['rank']!=784:raise RuntimeError('TRAIN span cannot identify the Linear function')
    folded=(w.T@transform).float();parent.shared_head.weight.copy_(folded)
    if upstream_state(parent)!=before:raise RuntimeError('Frozen upstream changed')
    candidate=evaluate(parent.shared_head,x,y,tasks,te)
    if whole_stage_selection(candidate,floors)<=whole_stage_selection(baseline,floors):raise RuntimeError('Recovered candidate does not improve parent; do not replace')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        parent_SHA256=a.parent_sha256,assistant_SHA256=sha(path),capture_features_SHA256=cr['features_sha256'],assistant_protocol_SHA256=sha(a.assistant_run/'protocol.json'),
        original_assistant_quality_gate=ar['assistant_quality_gate_passed'],absolute_admission_not_relaxed=True,final_student_fit='Original TRAIN and8 weak augmentations only; no EVAL input or target in function fit',
        TRAIN_sources={n:ids[tr][tasks[tr]==n].tolist() for n in set(tasks[tr])},TRAIN_query_rank=rank,original_student_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,**DISCLOSURE))
    for role in ('best','last'):
        saved=copy.deepcopy(cp);saved.update(model=parent.state_dict(),parent_checkpoint_sha256=a.parent_sha256,assistant_SHA256=sha(path),**DISCLOSURE);torch.save(saved,a.out/(role+'_checkpoint.pt'))
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,folded):raise RuntimeError('Strict CPU reload mismatch')
    put(a.out/'development.json',dict(scores=candidate,assistant=original,source_indices=ids[te].tolist(),tasks=tasks[te].tolist(),**DISCLOSURE))
    got=q.double()@folded.double().T
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),candidate=brief(candidate),absolute_goal_passed=within_bands(candidate),provisional_not_final=not within_bands(candidate),
        max_TRAIN_function_error=float((got-target).abs().max()),strict_CPU_reloaded=True,frozen_upstream=True,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),SDK=False,GPU=False,**DISCLOSURE))


if __name__=='__main__':main()
