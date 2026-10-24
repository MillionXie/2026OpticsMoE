"""Real-input CPU frontend contract for the selected laboratory .8655 body.

This is not a timing trial, instrumented-neural-stage contract, or GPU warmup.
Compare the original cached-input producer against live PNG/plain-token lookup
on real TEST examples outside the first200, using the unchanged original model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from protocol_identity import effective_protocol

TASK=REPO/'LightGenV2/tasks/t04_semantic_interaction'
BODY=TASK/'runs/simulation/layered_dc30_ccdsmall_selected_e45_s73_20260925/selected_checkpoint.pt'
PROFILE=TASK/'configs/layered_scene_exp05_dc30_ccdsmall.yaml'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
INFERENCE_FIELDS=('seed','image_size','grid_size','icon_classes','layout_version',
    'electronic_width','electronic_expansion','max_language_tokens','optical_fusion_initial',
    'embedding_only','shared_readout_enabled','shared_readout_variant','qwen_shared_baseline',
    'fusion_alpha_minimum','fusion_alpha_maximum','editor_depth','position_scale',
    'optical_shift_pixels','phase_dropout_p','zero_order_intensity_fraction',
    'ccd_noise_mean_fraction','ccd_noise_std_fraction','ccd_noise_min_fraction',
    'ccd_noise_max_fraction','amp_enabled','lightgen_model_variant')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def row_identity():
    row=next(r for r in effective_protocol()['rows'] if r['id']=='07')
    if (row['ours_epoch']!=45 or row['ours_lab_target']!=.8655 or row['ours_target']!=.8765
            or row['layout_version']!='layered_anchor6_svg_v3'):
        raise ValueError('Only the human-selected new-layout laboratory epoch45 identity')
    return row


def validate_settings(settings,original):
    differences={k:(getattr(settings,k),original.get(k)) for k in INFERENCE_FIELDS
                 if getattr(settings,k)!=original.get(k)}
    if differences:raise ValueError('Original inference settings changed: '+repr(differences))


def config_chain(path):
    import yaml
    path=Path(path).resolve();value=yaml.safe_load(path.read_text(encoding='utf8'))
    parents=value.get('base_config',[])
    if isinstance(parents,str):parents=[parents]
    result={path}
    for parent in parents:result.update(config_chain(path.parent/parent))
    return result


def source_files():
    files={Path(__file__),HERE/'protocol_identity.py',HERE/'protocol_amendment_openmoji_e45.json',
        HERE/'protocol.json',TASK/'layered_asset_identity_20261007.json'}
    files.update((TASK/'configs').glob('*.yaml'))
    for config in (PROFILE,REPO/'LightGenV2/tasks/t01_object_retrieval/configs/moe_optical_router_scale_matched_dc20.yaml',
            REPO/'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml'):
        files.update(config_chain(config))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_absolute():
            p=Path(name).resolve()
            if p.is_relative_to(REPO) and p.suffix=='.py':files.add(p)
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(files)}


def prepare(output):
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable task CPU contract required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only environment required')
    row=row_identity();original=json.loads((BODY.parent/'resolved_config.json').read_text())
    if sha(BODY)!=row['ours_pt_sha256']:raise ValueError('Wrong selected epoch45 body')
    output.mkdir(parents=True);write(output/'status.json',dict(status='loading',pid=os.getpid(),no_gpu=True))
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from torch import nn
    from PIL import Image
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.training import _set_phase_dropout
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import OpenMojiEditingDataset,load_prompt_cache
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import compose_prediction
    torch.set_num_threads(8);torch.manual_seed(73)
    try:
        settings=load_settings(PROFILE);validate_settings(settings,original)
        for key in ('data_dir','svg_asset_dir','asset_dir','qwen_checkpoint','prompt_cache_path'):
            setattr(settings,key,Path(original[key]))
        settings.output_dir=output;settings.num_workers=0
        # Rebind only the location of an identical original source config.
        if sha(settings.optical_base_config)!=sha(original['optical_base_config']):
            raise ValueError('Original optical base configuration is not identical on main')
        assets=json.loads((TASK/'layered_asset_identity_20261007.json').read_text())
        for name,record in assets['dataset_files'].items():
            if sha(settings.data_dir/name)!=record['sha256']:raise ValueError('Original dataset/cache SHA differs: '+name)
        if sha(settings.test_manifest)!=row['test_jsonl_sha256']:raise ValueError('Not the approved new-layout TEST')
        token_meta=json.loads(settings.prompt_cache_path.with_suffix('.json').read_text())
        if (token_meta['type']!='qwen_token_lookup_only_v1' or token_meta['native_transformer_blocks']!=0
                or token_meta['instruction_format']!='plain text, no chat wrapper'
                or token_meta['test_sha256']!=row['test_jsonl_sha256'] or token_meta['max_tokens']!=64):
            raise ValueError('Original producer was not the selected plain-token lookup frontend')
        checkpoint=settings.qwen_checkpoint
        weight_files=list(checkpoint.glob('*.safetensors'))
        if len(weight_files)!=1 or sha(weight_files[0])!=token_meta['weight_file_sha256']:
            raise ValueError('Original native Instruct weight-file identity changed')
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        with safe_open(str(weight_files[0]),framework='pt',device='cpu') as handle:
            weight=handle.get_tensor(token_meta['weight_key']).to(torch.bfloat16)
        if weight.ndim!=2 or weight.shape[1]!=2048:raise ValueError('Wrong original token embedding width')
        embedding=nn.Embedding.from_pretrained(weight,freeze=True).eval()
        payload=torch.load(BODY,map_location='cpu',weights_only=False)
        model=build_model(settings,torch.device('cpu'))
        if payload['epoch']!=45 or payload['architecture']!=model.checkpoint_architecture:
            raise ValueError('Original selected model architecture/epoch changed')
        model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
        _set_phase_dropout(model,False)
        prompts=load_prompt_cache(settings.prompt_cache_path)
        dataset=OpenMojiEditingDataset(settings.test_manifest,settings,prompts)
        if len(dataset)!=1000:raise ValueError('Complete original TEST is not1000')
        frozen=source_files();checks=[]
        with torch.inference_mode():
            for index in (200,999):
                original_input=dataset[index];record=dataset.records[index]
                expected=model(original_input['source_image'][None],[original_input['prompt_hidden']])
                # Same original file decode / float32 division. Online input
                # receives no cached feature and never supplies target labels.
                image_path=settings.data_dir/record['relative_dir']/'source.png'
                with Image.open(image_path) as image:
                    rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
                image_tensor=torch.from_numpy(rgb).permute(2,0,1).div(255.)
                ids=tokenizer.encode(record['instruction'],add_special_tokens=False)
                if not 0<len(ids)<=64:raise ValueError('Original instruction exceeds token budget')
                lookup=embedding(torch.tensor(ids,dtype=torch.long)).float()
                if not torch.equal(image_tensor,original_input['source_image']):raise ValueError('Original PNG frontend differs')
                if not torch.equal(lookup,original_input['prompt_hidden']):raise ValueError('Live tokenizer/embedding differs from original cached producer')
                actual=model(image_tensor[None],[lookup])
                keys=('category_logits','edit_logits','task_logits','condition','spatial')
                if not all(torch.equal(expected[k],actual[k]) for k in keys):raise ValueError('Original/full live-input CPU NN is not bitwise equal')
                expected_grid=compose_prediction(expected['category_logits'],expected['edit_logits'],original_input['source_grid'][None])
                actual_grid=compose_prediction(actual['category_logits'],actual['edit_logits'],torch.tensor(record['source_grid'])[None])
                if not all(torch.equal(a,b) for a,b in zip(expected_grid,actual_grid)):raise ValueError('Original CPU output grid differs')
                checks.append(dict(test_index=index,sample_id=record['sample_id'],source_png_sha256=sha(image_path),
                    image_tensor_shape=list(image_tensor.shape),plain_input_ids=ids,raw_lookup_dtype='bfloat16',
                    prompt_shape=list(lookup.shape),all_input_and_output_tensors_bitwise_equal=True,
                    output_shapes={k:list(actual[k].shape) for k in keys}))
        if torch.cuda.is_initialized():raise ValueError('CPU frontend contract created CUDA')
        if source_files()!=frozen:raise ValueError('Frozen original CPU source changed')
        result=dict(status='passed',no_gpu=True,real_test_examples=2,checks=checks,
            effective_identity=row,sources=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            selected_body_sha256=sha(BODY),original_token_producer_meta=token_meta,
            original_resolved_config_sha256=sha(BODY.parent/'resolved_config.json'),
            tokenizer_files={p.name:sha(p) for p in sorted(checkpoint.iterdir())
                if p.is_file() and p.suffix in ('.json','.jinja')},
            original_model_architecture=model.checkpoint_architecture,
            inference_settings={k:getattr(settings,k) for k in INFERENCE_FIELDS},
            selected_NN_stage_instrumentation_not_yet_tested=True,
            not_a_new_timing_or_energy_or_full_TEST_result=True)
        write(output/'contract.json',result);write(output/'status.json',dict(status='complete',pid=os.getpid(),no_gpu=True,real_test_examples=2))
        print(json.dumps(dict(status='passed',real_test_examples=2,no_gpu=True)),flush=True)
    except BaseException as error:
        write(output/'status.json',dict(status='failed',pid=os.getpid(),no_gpu=True,error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
