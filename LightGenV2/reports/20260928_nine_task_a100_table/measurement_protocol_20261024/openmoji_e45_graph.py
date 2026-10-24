"""Original epoch45 live frontend and non-substituting neural-stage observation.

The completed cached/live-input contract is reused as evidence, not repeated.
The executable CPU mode tests a DIFFERENT property: unchanged full live-input
forward versus the instrumented forward, on real TEST200/999. No timing trial,
CUDA, dummy image, cached online feature, Identity or synthetic CCD is used.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace,MethodType

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from prepare_openmoji_e45 import TASK,BODY,PROFILE,GPU,sha,write,row_identity,validate_settings,source_files
from stage_meter import StageMeter,statements,assign_index,source_record

FRONTEND_CONTRACT=TASK/'runs/simulation/a100_e45_live_frontend_contract_20261025_04/original_live_input_CPU_contract/contract.json'
FRONTEND_CONTRACT_SHA='ee9d944958ecab3d5eb44d05864ee25670d9c89197d37b6518c3b459b9dd628f'


def one(items,predicate):
    matches=[i for i,s in enumerate(items) if predicate(s)]
    if len(matches)!=1:raise ValueError('Exactly one original statement required: '+repr(matches))
    return matches[0]


def full_aperture_dispatch_contract(function):
    """Only this exact original super-call is adapted for instance-local spans."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef)).body
    expected=ast.parse('return super().forward(full_aperture_language_amplitude(input_fields))').body
    if ast.dump(ast.Module(body=body,type_ignores=[]))!=ast.dump(ast.Module(body=expected,type_ignores=[])):
        raise ValueError('Original full-aperture subclass dispatch changed; do not bypass its operations')


class AppliedStages:
    def __init__(self,meter):self.meter=meter;self.restores=[]
    def close(self):
        self.meter.close()
        for owner,name,original in reversed(self.restores):setattr(owner,name,original)
        self.restores.clear()


def patch_model(model,embedding,meter):
    from LightGenV2.tasks.t04_semantic_interaction.router_repair import (
        FullApertureLanguageRouter,full_aperture_language_amplitude)
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import OpticalDetectorTopKRouter
    applied=AppliedStages(meter)
    meter.wrap(embedding,'forward','frontend.plain_token_embedding',family='frontend')
    # Only the actual normalize/Conv3d/position expression is charged; temporal
    # duplication and block-major arrangement remain separate original layout.
    stem=model.vision_stem;body=statements(stem,'forward')
    start=assign_index(body,'normalized')
    grid=[i for i,s in enumerate(body) if s.startswith('grid = ')]
    if len(grid)!=2:raise ValueError('Original patch/position stem statements changed')
    meter.patch_statements(stem,'forward',[
        ('frontend.image_normalization','frontend',True,start,start),
        ('frontend.patch_and_position','frontend',True,grid[0],grid[1])])
    meter.wrap(stem,'_to_block_major','frontend.block_major_layout',family='layout',charged=False)
    body=statements(model,'_language_condition')
    first=assign_index(body,'positioned')
    last=one(body,lambda s:s.startswith('for group in groups:'))
    meter.patch_statements(model,'_language_condition',[
        ('frontend.original_sinusoidal_position','frontend',True,first,last)])
    body=statements(model,'forward')
    ix=one(body,lambda s:s.startswith('visual = visual +'))
    meter.patch_statements(model,'forward',[
        ('bridge.original_prompt_condition_to_vision','bridge',True,ix,ix)])
    meter.wrap(model.shared_readout,'summarize','head.position_linear_language_summary',family='head')
    meter.wrap(model.shared_readout,'forward','head.original_shared_grid_readout',family='head')
    for mode,core in (('language',model.language_core),('vision',model.vision_core)):
        if hasattr(core,'output_adapter'):raise ValueError('This epoch45 embedding-only core must not have a legacy output adapter')
        for name in ('input_adapter','input_norm','output_norm'):
            meter.wrap(getattr(core,name),'forward',mode+'.'+name,family='adapter')
        meter.wrap(core,'_fuse',mode+'.fusion',family='fusion',charged=False)
        for index,block in enumerate(core.blocks):
            meter.wrap(block,'forward',f'{mode}.parallel_residual{index+1}',family='parallel_residual',charged=False)
        branch=core.optical_branch;optical=branch.core;router=optical.router
        body=statements(optical,'encode_groups');ix=assign_index(body,'projected')
        meter.patch_statements(optical,'encode_groups',[(mode+'.amplitude_linear_norm_softplus','amplitude_encoding_nn',True,ix,ix)])
        body=statements(branch,'_encode_input_fields');ix=assign_index(body,'rms')
        meter.patch_statements(branch,'_encode_input_fields',[(mode+'.amplitude_rms','amplitude_encoding_nn',True,ix,len(body)-1)])
        meter.wrap(optical,'_direct_amplitude_load',mode+'.field_layout',family='layout',charged=False)
        meter.wrap(router,'_simulate',mode+'.router_simulation',family='simulation_only',charged=False)
        target=router
        if mode=='language':
            if type(router) is not FullApertureLanguageRouter:raise ValueError('Wrong original language router subclass')
            full_aperture_dispatch_contract(type(router).forward)
            # A proxy patches ONLY the unchanged bound parent method. The
            # subclass's original amplitude expansion still executes exactly
            # once. No global class mutation or zero-argument-super AST rewrite.
            target=SimpleNamespace(forward=MethodType(OpticalDetectorTopKRouter.forward,router))
            original=router.forward
            def measured(input_fields,*,_target=target):
                with meter.scope('language.full_aperture_amplitude_layout','layout',False):
                    expanded=full_aperture_language_amplitude(input_fields)
                return _target.forward(expanded)
            router.forward=measured;applied.restores.append((router,'forward',original))
            meter.ledger.append(source_record(original,'language.original_full_aperture_dispatch','layout',False))
            meter.ledger.append(source_record(full_aperture_language_amplitude,
                'language.full_aperture_amplitude_layout','layout',False))
        elif type(router) is not OpticalDetectorTopKRouter:raise ValueError('Wrong original visual router')
        body=statements(target,'forward');first=assign_index(body,'energy')
        last=one(body,lambda s:s.startswith('weights, selected, indices ='))
        meter.patch_statements(target,'forward',[(mode+'.router_energy_top2','router',True,first,last)])
        meter.wrap(branch,'_simulate_detector_roi',mode+'.expert_global_simulation',family='simulation_only',charged=False)
        meter.wrap(branch.ccd_normalizer,'forward',mode+'.original_scalar_ccd_normalization',family='ccd_nonlinearity')
        for name,readout,output in (('expert',branch.expert_readout,branch.expert_output_adapter),
                                   ('global',optical.readout,optical.output_adapter)):
            if type(readout).__name__!='PoolOnlyReadout':raise ValueError('Wrong original epoch45 CCD readout')
            meter.wrap(readout,'forward_intensity',f'{mode}.{name}_original_ccd_pool',family='ccd_readout')
            meter.wrap(output,'forward',f'{mode}.{name}_ccd_output_linear',family='ccd_readout')
    return applied


def tensor_shapes(value):
    if hasattr(value,'shape') and hasattr(value,'dtype'):
        return dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,(list,tuple)):return [tensor_shapes(v) for v in value]
    if isinstance(value,dict):return {k:tensor_shapes(v) for k,v in value.items()}
    return None


def shape_observers(model,embedding,rows,sample):
    handles=[]
    for prefix,root in (('model',model),('frontend_embedding',embedding)):
        for name,module in root.named_modules():
            if list(module.children()):continue
            def hook(owner,args,output,*,_name=prefix+('.'+name if name else '')):
                rows.append(dict(sample_index=sample(),module=_name,class_name=type(owner).__name__,
                    inputs=tensor_shapes(args),outputs=tensor_shapes(output)))
            handles.append(module.register_forward_hook(hook))
    return handles


def validate_stage_counts(counts,calls=1):
    expected={'frontend.plain_token_embedding':calls,'frontend.image_normalization':calls,
        'frontend.patch_and_position':calls,'frontend.original_sinusoidal_position':calls,
        'bridge.original_prompt_condition_to_vision':calls,'head.position_linear_language_summary':calls,
        'head.original_shared_grid_readout':calls,'language.full_aperture_amplitude_layout':calls}
    for mode in ('language','vision'):
        expected.update({mode+'.router_simulation':calls,mode+'.router_energy_top2':calls,
            mode+'.expert_global_simulation':2*calls,mode+'.original_scalar_ccd_normalization':2*calls,
            mode+'.expert_original_ccd_pool':calls,mode+'.global_original_ccd_pool':calls,
            mode+'.parallel_residual1':calls,mode+'.parallel_residual2':calls})
    if any(counts[k]!=v for k,v in expected.items()):raise ValueError('Original NN / six optical pass occurrence contract differs')
    return expected


def load_runtime(output,device):
    import torch
    from torch import nn
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.training import _set_phase_dropout
    proof=json.loads(FRONTEND_CONTRACT.read_text())
    if sha(FRONTEND_CONTRACT)!=FRONTEND_CONTRACT_SHA or proof['status']!='passed' or not proof['no_gpu']:
        raise ValueError('Reuse only the passed exact original live-input CPU contract')
    if any(sha(REPO/k)!=v for k,v in proof['sources'].items()):raise ValueError('CPU-bound original inference/config sources changed')
    original=json.loads((BODY.parent/'resolved_config.json').read_text());row=row_identity()
    if sha(BODY)!=row['ours_pt_sha256']:raise ValueError('Wrong adopted epoch45 body')
    settings=load_settings(PROFILE);validate_settings(settings,original)
    for key in ('data_dir','svg_asset_dir','asset_dir','qwen_checkpoint','prompt_cache_path'):setattr(settings,key,Path(original[key]))
    settings.output_dir=output;settings.num_workers=0
    if sha(settings.test_manifest)!=row['test_jsonl_sha256']:raise ValueError('Wrong original new-layout TEST')
    checkpoint=settings.qwen_checkpoint;weight_files=list(checkpoint.glob('*.safetensors'))
    meta=proof['original_token_producer_meta']
    if len(weight_files)!=1 or sha(weight_files[0])!=meta['weight_file_sha256']:raise ValueError('Original embedding/vision native file changed')
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    with safe_open(str(weight_files[0]),framework='pt',device='cpu') as values:
        weight=values.get_tensor(meta['weight_key']).to(torch.bfloat16)
    embedding=nn.Embedding.from_pretrained(weight,freeze=True).to(device).eval()
    torch.manual_seed(73);model=build_model(settings,torch.device('cpu'))
    payload=torch.load(BODY,map_location='cpu',weights_only=False)
    if payload['epoch']!=45 or payload['architecture']!=model.checkpoint_architecture:raise ValueError('Original epoch45 architecture differs')
    model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False);_set_phase_dropout(model,False);model.to(device)
    records=[json.loads(line) for line in settings.test_manifest.read_text().splitlines()]
    if len(records)!=1000:raise ValueError('Original TEST must have1000 examples')
    return model,embedding,tokenizer,settings,records,dict(frontend_contract=str(FRONTEND_CONTRACT),
        frontend_contract_sha256=FRONTEND_CONTRACT_SHA,selected_body_sha256=row['ours_pt_sha256'],effective_identity=row,
        native_weight_file_sha256=meta['weight_file_sha256'],original_architecture=model.checkpoint_architecture)


def live_input(record,settings,tokenizer,embedding,device):
    import torch
    import numpy as np
    from PIL import Image
    path=settings.data_dir/record['relative_dir']/record['files']['source']
    with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    pixels=torch.from_numpy(rgb).permute(2,0,1).div(255.)[None].to(device)
    ids=tokenizer.encode(record['instruction'],add_special_tokens=False)
    if not 0<len(ids)<=64:raise ValueError('No original token truncation permitted')
    hidden=embedding(torch.tensor(ids,dtype=torch.long,device=device)).float()
    return pixels,[hidden],ids


def cpu_contract(output):
    if output.exists() or not output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable task contract required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only contract environment required')
    output.mkdir(parents=True);write(output/'status.json',dict(status='loading',pid=os.getpid(),no_gpu=True))
    import torch
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import compose_prediction
    torch.set_num_threads(8);meter=StageMeter();applied=None;handles=[];shapes=[];checks=[]
    try:
        model,embedding,tokenizer,settings,records,identity=load_runtime(output,'cpu')
        frozen=source_files()
        with torch.inference_mode():
            for index in (200,999):
                if applied is not None:applied.close()
                for handle in handles:handle.remove()
                handles=[];row=records[index]
                pixels,hidden,ids=live_input(row,settings,tokenizer,embedding,'cpu')
                expected=model(pixels,hidden)
                applied=patch_model(model,embedding,meter);meter.sample_index=index
                handles=shape_observers(model,embedding,shapes,lambda:meter.sample_index)
                actual_pixels,actual_hidden,actual_ids=live_input(row,settings,tokenizer,embedding,'cpu')
                actual=model(actual_pixels,actual_hidden)
                keys=('category_logits','edit_logits','task_logits','condition','spatial')
                if actual_ids!=ids or not torch.equal(pixels,actual_pixels) or not torch.equal(hidden[0],actual_hidden[0]):
                    raise ValueError('Inserted stage observation changed actual live inputs')
                if not all(torch.equal(expected[k],actual[k]) for k in keys):raise ValueError('Inserted stages changed original model outputs')
                source=torch.tensor(row['source_grid'])[None]
                gold=compose_prediction(expected['category_logits'],expected['edit_logits'],source)
                pred=compose_prediction(actual['category_logits'],actual['edit_logits'],source)
                if not all(torch.equal(a,b) for a,b in zip(gold,pred)):raise ValueError('Original grid composition changed')
                counts=Counter(r['stage'] for r in meter.rows if r['sample_index']==index);validate_stage_counts(counts)
                checks.append(dict(test_index=index,sample_id=row['sample_id'],bitwise_equal=True,
                    source_png_sha256=sha(settings.data_dir/row['relative_dir']/row['files']['source']),
                    input_shapes=tensor_shapes((actual_pixels,actual_hidden)),output_shapes={k:list(actual[k].shape) for k in keys},
                    observed_stage_counts=dict(counts)))
        if torch.cuda.is_initialized():raise ValueError('CPU stage contract initialized CUDA')
        if source_files()!=frozen:raise ValueError('Frozen original stage-contract sources changed')
        unique={json.dumps(r,sort_keys=True):r for r in meter.ledger}
        write(output/'contract.json',dict(status='passed',no_gpu=True,real_test_examples=2,identity=identity,checks=checks,
            stage_ledger=list(unique.values()),nn_module_shapes=shapes,sources=frozen,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            cached_live_frontend_contract_reused_not_retested=True,not_a_formal_timing_or_energy_or_full_TEST_result=True))
        write(output/'status.json',dict(status='complete',pid=os.getpid(),real_test_examples=2,no_gpu=True))
        print(json.dumps(dict(status='passed',real_test_examples=2,no_gpu=True)))
    except BaseException as error:
        write(output/'status.json',dict(status='failed',pid=os.getpid(),error=repr(error),no_gpu=True));raise
    finally:
        for handle in handles:handle.remove()
        if applied is not None:applied.close()
        else:meter.close()
        if not (output/'contract.json').exists():
            write(output/'rejected_stage_observations.json',dict(checks=checks,stages=meter.rows,
                stage_ledger=meter.ledger,nn_module_shapes=shapes,not_an_accepted_contract=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    cpu_contract(parser.parse_args().output.resolve())
