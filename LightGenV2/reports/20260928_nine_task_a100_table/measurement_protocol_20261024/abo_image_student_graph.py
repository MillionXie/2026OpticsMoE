"""Original .8058 image-query student; reuse existing span observers, no training.

The archived optimized profile differs from its balance1 base in training and
asset locations only. Check that explicitly, load its exact PT strictly, and
compare original real-image/title forwards before any formal input-cost trial.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import TASK,MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,sha,write,model_assets
from abo_student_graph import patch

BODY_SHA='a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'
PROFILE=TASK/'configs/optical_router_moe_dc20_kd1_balance1.yaml'
ARCHIVE=REPO/'archive/legacy_lab_projects/ABO_Lab_8um/original_inference'
ARCHIVED_PROFILE=ARCHIVE/'backend/LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_router_moe_dc20_kd1_balance1_optimized.yaml'
ARCHIVED_PROFILE_SHA='146ad18d75b1f58ab0df8204039fef46626fb55f3b42c50f6e3e9e54986022f7'
BODY=ARCHIVE/'assets/best_checkpoint.pt'


def inference_signature(raw):
    value=copy.deepcopy(raw)
    for key in ('output_dir','training_optimization'):value.pop(key,None)
    for section,key in (('qwen','model_id'),('teacher_cache','directory'),('abo_image_text','cache_file')):
        value.get(section,{}).pop(key,None)
    for key in ('epochs','learning_rate','adapter_learning_rate','readout_learning_rate',
                'phase_learning_rate','router_learning_rate','ema_decay','lambda_router_balance',
                'lambda_router_importance','lambda_router_hard_load_balance'):
        value.get('training',{}).pop(key,None)
    return value


def imported_sources():
    paths={Path(__file__),HERE/'abo_student_graph.py',PROFILE}
    from prepare_openmoji_e45 import config_chain
    paths.update(config_chain(PROFILE))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_absolute():
            p=Path(name).resolve()
            if p.is_relative_to(REPO) and p.suffix=='.py':paths.add(p)
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}


def load(args,device):
    import torch
    import yaml
    from LightGenV2.tasks.t08_abo_image_text_retrieval import optical_moe as original
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import _read_config
    if sha(args.checkpoint)!=BODY_SHA or sha(ARCHIVED_PROFILE)!=ARCHIVED_PROFILE_SHA:
        raise ValueError('Only exact .8058 EMA body and original optimized profile')
    if sha(args.model/'config.json')!=MODEL_CONFIG_SHA or sha(args.model/'model.safetensors')!=MODEL_WEIGHTS_SHA:
        raise ValueError('Original frozen Embedding2B identity differs')
    archive_raw=_read_config(ARCHIVED_PROFILE);base_raw=_read_config(PROFILE)
    if inference_signature(archive_raw)!=inference_signature(base_raw):
        raise ValueError('Original optimized profile inference differs from audited balance1 base')
    raw=copy.deepcopy(archive_raw)
    # Preserve inherited locations; none of the teacher/warmstart PTs is loaded
    # by this strict fixed-state online query entry.
    for section,key in (('qwen','cache_dir'),('teacher_cache','directory'),('router_experiment','source_checkpoint'),
            ('hybrid','initial_electronic_checkpoint'),('abo_image_text','cache_file')):
        value=raw.get(section,{}).get(key)
        if value is not None:
            p=Path(value);raw[section][key]=str(p if p.is_absolute() else (ARCHIVED_PROFILE.parent/p).resolve())
    for key,value in raw.get('warmstart',{}).items():
        if key.endswith('_checkpoint') and value is not None:
            p=Path(value);raw['warmstart'][key]=str(p if p.is_absolute() else (ARCHIVED_PROFILE.parent/p).resolve())
    raw['qwen']['model_id']=str(args.model);raw['qwen']['local_files_only']=True
    raw['dataset']['dataset_root']=str(args.data);raw['output_dir']=str(args.output)
    config=args.output/'resolved_input.yaml';config.write_text(yaml.safe_dump(raw,sort_keys=False),encoding='utf8')
    settings=original.load_settings(config);settings.inference_batch_size=1;settings.num_workers=0
    settings.router_optimization_seed=42;settings.random_seed=42;original.seed_everything(42)
    contract=original.load_contract(args.data)
    loaded=original.load_backbone(settings,torch.device(device));replacement,readout=original.build_student(loaded,settings)
    payload=original.load_checkpoint(args.checkpoint,replacement,readout)
    if payload.get('epoch')!=25 or payload.get('metadata',{}).get('weight_variant')!='ema':
        raise ValueError('Not the adopted epoch25 EMA state')
    loaded.model.eval().requires_grad_(False);replacement.use_student()
    replacement.vision_surrogate.eval().requires_grad_(False);replacement.language_surrogate.eval().requires_grad_(False)
    replacement.set_phase_dropout_active(False);readout.eval().requires_grad_(False)
    info=dict(body_sha256=BODY_SHA,original_optimized_profile_sha256=ARCHIVED_PROFILE_SHA,
        model_config_sha256=MODEL_CONFIG_SHA,model_weights_sha256=MODEL_WEIGHTS_SHA,
        model_assets=model_assets(args.model),epoch=25,weight_variant='ema',
        full2400_confirmed_reference_hit1=.8058333333333333,
        historical_same_PT_A100_replay_hit1=.8045833333333333,
        resolved_settings=json.loads(json.dumps(vars(settings),default=str)),dataset_sha256=contract.sha256,
        original_module_sha256=sha(TASK/'optical_moe.py'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    return loaded,replacement,readout,settings,contract,info


def image_forward(loaded,replacement,readout,settings,sample):
    import torch
    from LightGenV2.tasks.t08_abo_image_text_retrieval import optical_moe as original
    image=original.GroceryRetrievalDataset([sample],settings.image_size,augment=False)[0]['image']
    inputs=original.preprocess_images(loaded.processor,[image],settings.instruction)
    original.validate_token_budgets(inputs,settings)
    moved=original.move_inputs(inputs,loaded.device)
    with torch.autocast(device_type=loaded.device.type,
            dtype=torch.bfloat16 if settings.dtype=='bfloat16' else torch.float16,
            enabled=settings.amp_enabled and loaded.device.type=='cuda'):
        embedding,features=original.student_embeddings(loaded.model,replacement,readout,moved)
    return embedding,features,inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model','data','checkpoint','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable task CPU contract required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only process required')
    args.output.mkdir(parents=True);write(args.output/'status.json',dict(status='loading',pid=os.getpid(),no_gpu=True))
    import torch
    from collections import Counter
    from stage_meter import StageMeter
    from LightGenV2.tasks.t08_abo_image_text_retrieval import optical_moe as original
    torch.set_num_threads(8);meter=StageMeter();replacement=None
    try:
        loaded,replacement,readout,settings,contract,info=load(args,'cpu');frozen=imported_sources();checks=[]
        with torch.inference_mode():
            for index in (200,2399):
                meter.close();sample=contract.test[index]
                expected=original.encode_student_samples(loaded,replacement,readout,[sample],settings)
                patch(loaded,replacement,readout,meter);meter.sample_index=index
                actual,features,inputs=image_forward(loaded,replacement,readout,settings,sample)
                if not torch.equal(expected,actual.cpu()):raise ValueError('Real original/instrumented image forward differs')
                counts=Counter(r['stage'] for r in meter.rows if r['sample_index']==index)
                if any(counts[m+'.router_simulation']!=1 or counts[m+'.expert_global_simulation']!=2 for m in ('vision','language')):
                    raise ValueError('Actual image query must preserve six optical passes')
                checks.append(dict(kind='image',test_index=index,sample_id=sample.sample_id,bitwise_equal=True,
                    input_shapes={k:list(v.shape) for k,v in inputs.items()},embedding_shape=list(actual.shape),
                    feature_shape=list(features.shape),stage_counts=dict(counts)))
            for index in (0,99):
                meter.close();title=contract.titles[index]
                expected=original._encode_titles_student(loaded,replacement,readout,[title],settings)
                patch(loaded,replacement,readout,meter);meter.sample_index=2400+index
                actual=original._encode_titles_student(loaded,replacement,readout,[title],settings)
                if not torch.equal(expected,actual):raise ValueError('Original/instrumented title bank forward differs')
                checks.append(dict(kind='title',title_index=index,product_id=title.product_id,bitwise_equal=True))
        if imported_sources()!=frozen or torch.cuda.is_initialized():raise ValueError('Frozen sources or CPU-only contract changed')
        write(args.output/'contract.json',dict(status='passed',no_gpu=True,checks=checks,identity=info,
            stage_ledger=meter.ledger,sources=frozen,formal_first200_not_warmed=True,
            not_a_formal_timing_or_energy_run=True))
        write(args.output/'status.json',dict(status='complete',pid=os.getpid(),no_gpu=True,real_images=2,real_titles=2))
    except BaseException as error:write(args.output/'status.json',dict(status='failed',error=repr(error),no_gpu=True));raise
    finally:
        meter.close()
        if replacement is not None:replacement.close()


if __name__=='__main__':main()
