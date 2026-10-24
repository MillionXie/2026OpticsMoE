"""Adopted native Qwen producer plus original shared head, never cached E2E.

The original extract_native_one performs native PNG/chat processing, every
native vision/language layer, pre-merger capture and BF16 CPU feature returns.
Keep those original transfers and the original head's float32/H2D conversions;
do not silently substitute a faster GPU-only path or its Identity cache slot.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from prepare_openmoji_e45 import sha,write,source_files,config_chain
from protocol_identity import effective_protocol

TASK=REPO/'LightGenV2/tasks/t04_semantic_interaction'
PROFILE=TASK/'configs/layered_scene_qwen_shared.yaml'
ORIGINAL=TASK/'runs/simulation/layered_scene_qwen_shared_s73_e100_20260920'
CONFIG_SHA='520930847865400eabf39c9420cf131bf787ec18cf41a92c73973923654034c4'
REFERENCE_SHA='dd22cbdc39e5e6b77606ee11b45c1dbcf49e5d4dd7791485afbb56870c1482f1'
NATIVE_SHA='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def validate_identity(row,original,reference):
    if (row['qwen_target']!=.812 or row['qwen_epoch']!=30
            or row['layout_version']!='layered_anchor6_svg_v3'
            or original['lightgen_model_variant']!='qwen_frozen_shared_readout'
            or original['embedding_only'] or not original['qwen_shared_baseline']
            or original['image_size']!=224 or original['grid_size']!=6
            or not original['amp_enabled'] or reference['selected_epoch']!=30
            or reference['test_samples']!=1000
            or reference['metrics']['overall']['changed_cell_accuracy']!=.812):
        raise ValueError('Not the adopted same-layout full-native epoch30 baseline')


class ObserveProcessor:
    """Return the exact original BatchEncoding unchanged; metadata observation."""
    def __init__(self,processor):self.processor=processor;self.records=[]
    def __getattr__(self,name):return getattr(self.processor,name)
    def __call__(self,*args,**kwargs):
        inputs=self.processor(*args,**kwargs)
        self.records.append(dict(input_ids=inputs['input_ids'].tolist(),
            attention_mask=inputs['attention_mask'].tolist(),
            shapes={k:list(v.shape) for k,v in inputs.items()},
            dtypes={k:str(v.dtype) for k,v in inputs.items()}))
        return inputs


def native_and_head(runtime,row,device):
    import torch
    from LightGenV2.tasks.t04_semantic_interaction.qwen_shared import extract_native_one
    settings,native,processor,head=runtime
    observed=ObserveProcessor(processor)
    visual,language,counts=extract_native_one(native,observed,row,settings,device)
    if len(observed.records)!=1:raise ValueError('Original native processor call count changed')
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=settings.amp_enabled):
        outputs=head(visual.unsqueeze(0),[language])
    return outputs,observed.records[0],counts,(visual,language)


def load_runtime(output,device):
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    original=json.loads((ORIGINAL/'resolved_config.json').read_text())
    reference=json.loads((ORIGINAL/'selected_checkpoint_test_evaluation.json').read_text())
    row=next(r for r in effective_protocol()['rows'] if r['id']=='07')
    validate_identity(row,original,reference)
    if (sha(ORIGINAL/'resolved_config.json')!=CONFIG_SHA
            or sha(ORIGINAL/'selected_checkpoint_test_evaluation.json')!=REFERENCE_SHA
            or sha(ORIGINAL/'best_checkpoint.pt')!=row['qwen_pt_sha256']):
        raise ValueError('Original epoch30 body/config/performance identity differs')
    settings=load_settings(PROFILE)
    # Preserve the original stored location even for the unused training-only
    # legacy entry. build_model's Qwen branch never calls initialize_from_legacy;
    # its own early return also explicitly declares legacy_weights_loaded=False.
    paths=('data_dir','svg_asset_dir','asset_dir','qwen_checkpoint','prompt_cache_path',
           'legacy_warmstart_checkpoint')
    not_inference={'config_path','output_dir','optical_base_config',*paths}
    differences={k:(str(getattr(settings,k)),str(v)) for k,v in original.items()
        if k not in not_inference and hasattr(settings,k) and getattr(settings,k)!=v}
    if differences:raise ValueError('Original resolved inference/training fields changed: '+repr(differences))
    for key in paths:setattr(settings,key,Path(original[key]))
    if sha(settings.optical_base_config)!=sha(original['optical_base_config']):
        raise ValueError('Original shared-head base configuration differs')
    if sha(settings.test_manifest)!=row['test_jsonl_sha256']:raise ValueError('Not the exact matched new-layout TEST')
    weight=settings.qwen_checkpoint/'model.safetensors'
    if sha(weight)!=NATIVE_SHA:raise ValueError('Original complete native checkpoint differs')
    settings.output_dir=output;settings.num_workers=0
    # CPU loading of the original float32 trained head, strict exact state.
    head=build_model(settings,device)
    payload=torch.load(ORIGINAL/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    head.load_state_dict(payload['model'],strict=True);head.eval().requires_grad_(False)
    native=Qwen3VLForConditionalGeneration.from_pretrained(settings.qwen_checkpoint,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(settings.qwen_checkpoint,local_files_only=True,
        min_pixels=224**2,max_pixels=224**2)
    return (settings,native,processor,head),dict(selected_body_sha256=row['qwen_pt_sha256'],
        original_config_sha256=CONFIG_SHA,full1000_historical_reference_sha256=REFERENCE_SHA,
        native_weight_sha256=NATIVE_SHA,test_manifest_sha256=row['test_jsonl_sha256'])


def prepare(output):
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable task CPU native contract required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only environment required')
    output.mkdir(parents=True);write(output/'status.json',dict(status='loading',pid=os.getpid(),no_gpu=True))
    import torch
    from LightGenV2.tasks.t04_semantic_interaction.qwen_shared import extract_native_one
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import compose_prediction
    torch.set_num_threads(8);torch.manual_seed(73)
    try:
        device=torch.device('cpu');runtime,identity=load_runtime(output,device)
        settings,native,processor,head=runtime
        records=[json.loads(line) for line in settings.test_manifest.read_text().splitlines()]
        frozen=source_files();frozen[Path(__file__).relative_to(REPO).as_posix()]=sha(__file__)
        checks=[]
        with torch.inference_mode():
            for index in (200,999):
                row=records[index]
                write(output/'status.json',dict(status='original_and_observed_native_CPU_forward',test_index=index,pid=os.getpid(),no_gpu=True))
                # Original task producer and original strict head are the gold.
                visual,language,counts=extract_native_one(native,processor,row,settings,device)
                with torch.autocast(device_type='cpu',dtype=torch.bfloat16,enabled=settings.amp_enabled):
                    gold=head(visual.unsqueeze(0),[language])
                actual,inputs,actual_counts,features=native_and_head(runtime,row,device)
                if counts!=actual_counts or not torch.equal(visual,features[0]) or not torch.equal(language,features[1]):
                    raise ValueError('Observed native producer changed actual features/layer execution')
                if gold.keys()!=actual.keys() or any(not torch.equal(gold[k],actual[k]) for k in gold):
                    raise ValueError('Complete original shared-head CPU output not bitwise equal')
                source=torch.tensor(row['source_grid']).unsqueeze(0)
                expected=compose_prediction(gold,source);result=compose_prediction(actual,source)
                if any(not torch.equal(a,b) for a,b in zip(expected,result)):
                    raise ValueError('Original full CPU grid composition differs')
                valid=sum(inputs['attention_mask'][0])
                if valid!=len(inputs['input_ids'][0]):raise ValueError('Original B1 mask/padding differs')
                checks.append(dict(test_index=index,sample_id=row['sample_id'],bitwise_equal=True,
                    native_counts=counts,actual_input=inputs,actual_valid_input_tokens=valid,
                    visual_shape=list(visual.shape),language_shape=list(language.shape),
                    original_PNG_sha256=sha(settings.data_dir/row['relative_dir']/row['files']['source']),
                    target_not_provided_to_model=True))
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Original CPU contract source changed')
        if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA initialization')
        value=dict(status='passed',no_gpu=True,real_test_examples=2,checks=checks,identity=identity,
            sources=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            native_scope='original qwen_shared.extract_native_one including BF16 CPU feature return, then original head float32/H2D and original compose_prediction',
            training_cache_used_as_online_input=False,formal_first200_not_warmed=True,
            common_formal200_tokens_not_yet_measured=True,not_a_formal_timing_trial=True)
        write(output/'contract.json',value);write(output/'status.json',dict(status='complete',no_gpu=True,real_examples=2))
        print(json.dumps(dict(status='passed',real_examples=2,no_gpu=True,contract_sha256=sha(output/'contract.json'))),flush=True)
    except BaseException as error:
        write(output/'status.json',dict(status='failed',error=repr(error),no_gpu=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
