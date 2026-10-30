"""Explicit heldout DEV128 physical acquisition and complete feature replay.

No TEST/targets/gradients. Original F6/F9, pixels/prompts/seeds and physical
mapping remain fixed. Evaluation data is never relabeled as TRAIN.
"""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
from .scene_lab_protocol import Ledger,STAGES,identity,sha,put,NeedCapture
from .scene_lab_native import NativeRunner,load_head,F9,F6


def dev_requests(rows,train):
    if len(rows)!=128 or len({r['sample_id']for r in rows})!=128:raise ValueError('Require the complete fixed DEV128')
    if len({r['reference_sha256']for r in rows})!=16 or {r['category']for r in rows}!={'lamp','pillow'}:raise ValueError('Require original two-category/sixteen-reference DEV')
    if {r['reference_sha256']for r in rows}&{r['reference_sha256']for r in train}:raise ValueError('DEV source pixels overlap TRAIN')
    selected=[]
    for row in rows:
        if row['prompt_profile']!='two_sentence_natural_room_v4':raise ValueError('Preserve original natural prompts')
        for forbidden in ('target','mask','latent','feature'):
            if any(forbidden in name.lower()for name in row):raise ValueError('No target/mask/latent/feature inputs')
        selected.append(dict(sample_id=identity(dict(source_sample_id=row['sample_id'],reference_sha256=row['reference_sha256'],prompt=row['prompt'],seed=row['seed']))[:24],
            split='dev',reference=row['reference'],reference_sha256=row['reference_sha256'],prompt=row['prompt'],seed=row['seed'],category=row['category']))
    return selected


def initialize(a):
    c=json.loads((a.parent_ledger/'contract.json').read_text());rows=json.loads((a.dev/'inference_inputs.json').read_text());report=json.loads((a.dev/'report.json').read_text())
    if not report['complete'] or report['schema']!='scene_natural_prompt_DEV_v1' or report['request_sha256']!=sha(a.dev/'inference_inputs.json'):raise ValueError('Changed DEV protocol')
    requests=dev_requests(rows,c['requests'])
    for r in requests:
        path=(a.references/r['reference']).resolve()
        if not path.is_relative_to(a.references.resolve())or sha(path)!=r['reference_sha256']:raise ValueError('Changed DEV reference')
    session=dict(c['session'],session_id=c['session']['session_id']+'_DEV128',cohort_role='dev',evaluation_only=True,evaluation_manifest_sha256=report['request_sha256'])
    ledger=Ledger(a.ledger,session,dict(F9=F9,F6=F6,source_commit=a.source_commit),requests,create=True)
    put(a.ledger.parent/'evaluation_protocol.json',dict(schema='scene_lab_DEV128_protocol_v1',ledger_SHA256=ledger.key,TRAIN_disjoint=True,
        request_SHA256=report['request_sha256'],DEV=128,actual_CCD_required=768,F9=F9,F6=F6,deadline_Beijing='2026-10-31T09:00:00+08:00',
        roles='heldout development evaluation, no gradients; not independent TEST500',primary_quality_reference='original frozen F9 SIM, never a weaker substitute',
        metrics=['CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact'],groups=['all','lamp','pillow'],relative_drop_max=.05,
        relative_drop_formula='(SIM-EXP)/abs(SIM); near-zero SIM requires absolute difference',
        additional_reference='adapted-weight matched SIM is reported separately; cannot replace original F9 quality reference',
        scene_GT_or_masks_used=False,TEST_opened=False,calibration_unchanged=True))
    put(a.ledger.parent/'original_public_DEV_requests.json',rows)
    print(json.dumps(dict(DEV=128,stages=6,total_CCD=768,ledger_SHA256=ledger.key)),flush=True)


def run(a):
    import torch
    from .qwen3_scene_paired_train import new_qwen
    from .scene_paired_render import install_optical_frontend
    from .scene_lab_pilot import mark_input_ready
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or not a.gpu_uuid.startswith('GPU-'):raise ValueError('Explicit GPU identity required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4,0)
    c=json.loads((a.ledger/'contract.json').read_text());ledger=Ledger(a.ledger,c['session'],c['models'],c['requests'])
    if len(ledger.requests)!=128 or ledger.session.get('cohort_role')!='dev' or ledger.session.get('evaluation_only')is not True:raise ValueError('Explicit DEV-only ledger required')
    if a.mode=='prepare':admission=ledger.admit_stage(a.stage)
    else:
        for stage in STAGES:
            for request in ledger.requests:ledger.receipt(stage,request)
    task=Path(__file__).resolve().parent;base=task/'runs/simulation';qwen,processor=new_qwen(SimpleNamespace(qwen=a.qwen,qwen_side=256))
    optical=install_optical_frontend(qwen,base/'qwen3_scene_LG_frontend_20261029',base/'qwen3_scene_LG_teacher_features_20261029',base/'scene_natural_TRAIN608_condition_cache_20261030/LG')
    if optical['checkpoint_sha256']!=F6:raise ValueError('Wrong frontend')
    head,_=load_head(base/'scene_natural_single_EG_joint12_training_20261030/best_checkpoint.pt',expected_sha256=F9)
    runner=NativeRunner(qwen,processor,head.cuda(),a.references,frontend_sha256=F6,head_sha256=F9,export_features=a.mode=='features')
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'execution.json',dict(source_commit=a.source_commit,source_SHA256=sha(__file__),mode=a.mode,SDK=False,gradients=False))
    if a.mode=='prepare':
        count=0
        def streamed(request,provider):
            nonlocal count
            try:return runner(request,provider)
            except NeedCapture:
                mark_input_ready(ledger.folder(a.stage,request));count+=1
                if count%16==0:print(json.dumps(dict(stage=a.stage,DEV_prepared=count,total=128)),flush=True)
                time.sleep(.03);raise
        pending=ledger.prepare(a.stage,streamed,admission=admission)
        put(a.out/'report.json',dict(complete=True,DEV=128,stage=a.stage,pending_count=len(pending),ledger_SHA256=ledger.key,SDK=False,TEST_used=False,gradients=False))
    else:
        features={}
        for index,request in enumerate(ledger.requests):
            result=ledger.infer(request,runner);features[request['sample_id']]=result['features']
            if(index+1)%16==0:print(json.dumps(dict(DEV_replayed=index+1,total=128)),flush=True)
        torch.save(dict(schema='scene_lab_DEV_measured_feature_cache_v1',features=features,records=ledger.requests,F6=F6,F9=F9,ledger_SHA256=ledger.key,evaluation_only=True),a.out/'DEV_features.pt')
        put(a.out/'report.json',dict(complete=True,DEV=128,feature_cache_SHA256=sha(a.out/'DEV_features.pt'),all_six_stages_measured=True,SDK=False,TEST_used=False,gradients=False,quality_accepted=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('init','prepare','features'),required=True)
    for name in ('parent-ledger','ledger','dev','references','qwen','out'):p.add_argument('--'+name,type=Path)
    p.add_argument('--stage',choices=STAGES);p.add_argument('--source-commit',required=True);p.add_argument('--gpu-uuid')
    a=p.parse_args()
    if len(a.source_commit)!=40:raise ValueError('Published source required')
    if a.mode=='init':initialize(a)
    else:run(a)


if __name__=='__main__':main()
