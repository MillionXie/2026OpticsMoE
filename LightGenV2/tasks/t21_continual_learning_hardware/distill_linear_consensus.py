"""Fixed equal two-teacher TRAIN-query function recovery, no EVAL coefficient fit.

One original shared Linear student. Both teachers inherit partial TEST sources;
this is explicitly development adaptation, never independent evaluation.
"""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
import torch
from .audit_student_transfer import linear_projection
from .distill_capacity_matched_stage import TRAIN_queries
from .distill_margin_teacher import within_bands
from .fit_gated_linear_teacher import DISCLOSURE,full_covariance_transform
from .refine_measured_ce import contrast_transform
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import put,sha


def equal_consensus(query,heads):
    if len(heads)!=2 or heads[0].shape!=heads[1].shape:raise ValueError('Exactly two whole-stage heads required')
    target=sum(query.double()@w.detach().double().T for w in heads)/2
    return target-target.mean(1,keepdim=True)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','first_run','second_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--LG-old-new-consensus',action='store_true');p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Inherited TEST information must be disclosed')
    torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    architecture='moe' if a.LG_old_new_consensus else 'd2nn'
    if cp['architecture']!=architecture or cp['stage']!=2 or cp['measured_features_sha256']!=cr['features_sha256'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or not cr['passed'] or not cr['SDK_released'] or cr['pilot']:raise ValueError('Sealed measured B only')
    teachers=[];receipts=[];source_ids=None
    expected=('T21_shared_margin_teacher_BC_v1','T21_shared_bounded_margin_teacher_BC_v4')
    if a.LG_old_new_consensus:expected=('T21_LG_B_shared_margin_teacher_v1','T21_LG_B_new_task_teacher_old_TRAIN_function_v1')
    individual_sources=[]
    for index,(folder,schema) in enumerate(zip((a.first_run,a.second_run),expected)):
        protocol=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'report.json').read_text());path=folder/('last_linear_teacher_checkpoint.pt' if a.LG_old_new_consensus and index==1 else 'linear_teacher_checkpoint.pt')
        expected_parent=a.parent_sha256
        if a.LG_old_new_consensus and index==0:expected_parent='02d36c3197f64e6900ae8c13fdb60b6ad61a79239e4b6cf0f0269529f50b0962'
        if not report['passed'] or protocol['profile']['schema']!=schema or protocol['parent_SHA256']!=expected_parent or protocol['capture_features_SHA256']!=cr['features_sha256'] or report['valid_for_formal_test_table'] is not False:raise ValueError('Exact complementary fixed-scope teachers required')
        if a.LG_old_new_consensus:
            if {n:len(v) for n,v in protocol['selection_source_ids'].items()}!={n:500 if index==0 else 1000 for n in ('eurosat','clevr')}:raise ValueError('Disclosed LG source identity differs')
            individual_sources.append(protocol['selection_source_ids']);source_ids=protocol['selection_source_ids']
        else:
            if source_ids is None:source_ids=protocol['selection_source_ids']
            if source_ids!=protocol['selection_source_ids'] or {n:len(v) for n,v in source_ids.items()}!=dict(eurosat=200,clevr=800):raise ValueError('Partial source pool changed')
        teacher,_=load_checkpoint(path,sha(path))
        if upstream_state(teacher)!=before:raise ValueError('Teachers do not share frozen prefix')
        teachers.append(teacher.shared_head.weight);receipts.append(dict(path=str(path),SHA256=sha(path),protocol_SHA256=sha(folder/'protocol.json')))
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:x=torch.tensor(d['features']);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if len(tr)!=1300 or len(te)!=2000 or any(int((tasks[te]==n).sum())!=1000 for n in ('eurosat','clevr')):raise ValueError('Complete fixed B identities required')
    cfg=json.loads(Path(__file__).with_name('config_margin_teacher_20261025.json').read_text());query,qt=TRAIN_queries(x[tr],tasks[tr],cfg)
    transform,_=full_covariance_transform(x[tr],contrast_transform(x[tr]));target=equal_consensus(query,teachers);weight,rank=linear_projection(query.double()@transform.T,target,qt)
    if rank['rank']!=784:raise RuntimeError('TRAIN queries do not identify the function')
    folded=(weight.T@transform).float();model.shared_head.weight.copy_(folded)
    if upstream_state(model)!=before:raise RuntimeError('Frozen prefix changed')
    recovered=query.double()@folded.double().T
    if float((recovered.argmax(1)==target.argmax(1)).double().mean())!=1:raise RuntimeError('TRAIN function argmax not recovered')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),parent_SHA256=a.parent_sha256,capture_features_SHA256=cr['features_sha256'],teachers=receipts,selection_source_ids=source_ids,individual_teacher_source_ids=individual_sources,all_TEST_sources_used=a.LG_old_new_consensus,teacher_weights=[.5,.5],weights_fixed_before_EVAL=True,coefficient_scan=False,task_head_splicing=False,student_fit_EVAL_inputs=0,TRAIN_query_rank=rank,SDK=False,GPU=False,**DISCLOSURE))
    for role in ('best','last'):
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,folded):raise RuntimeError('Strict CPU reload mismatch')
    groups=dict(TRAIN=tr,all_development=te);scores={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in groups.items()}
    put(a.out/'development.json',dict(scores=scores,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE))
    put(a.out/'report.json',dict(passed=True,student={n:brief(v) for n,v in scores.items()},all_tasks_within_requested_band=within_bands(scores['all_development'],2,architecture),LG_CLEVR_at_least_current_D2_candidate=scores['all_development']['clevr']['correct']>=756 if a.LG_old_new_consensus else None,student_sha256=sha(a.out/'best_checkpoint.pt'),max_TRAIN_function_error=float((recovered-target).abs().max()),strict_CPU_reloaded=True,frozen_upstream=True,SDK=False,GPU=False,**DISCLOSURE))


if __name__=='__main__':main()
