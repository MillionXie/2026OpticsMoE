"""Original5b4f CPU input/stage contract; cache is reference only, never online input.

Recreate original reference composition and original chat tokenizer/embedding.
Observe original NN without replacing FFT, residuals or decoder. No training.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import sha,write
TASK=REPO/'LightGenV2/tasks/t12_text_to_image'
ASSETS=Path('/DATA/DATA1/guest3/t12_assets')
ORIGINAL_RUNS=REPO/'.worktrees/t12_physical_robust_v2_20260927/LightGenV2/tasks/t12_text_to_image/runs/simulation'
BODY=ORIGINAL_RUNS/'20260927_language_balance_input/last_checkpoint.pt'
BODY_SHA='5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac'
BASELINE=ORIGINAL_RUNS/'20260927_matched_qwen28_baseline/best_checkpoint.pt'
BASELINE_SHA='599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52'
MODEL=Path('/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203')
WEIGHT_SHA='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
PINS={'abo_unified_expanded_instructions_qwen2_v2.pt':'66f457115cc7b1058f3ee42be0fec5815f101780c22b4805ad08940966dae97f',
      'abo_unified_expanded_qwen_embeddings_v2.pt':'e7a855849ef22e470aafdcf8ee583968ca0c8d5d17ff42603b103c49e97502ac'}
DATA=ASSETS/'datasets/abo_cleanrender_lamp_table_pillow_256_v1'
TEST_SHA='331874abbb8d8c7a4ac5ee3d6e030f20611cb5a9e96d3092917b7a5c4d7d5d2f'


def reference_input(dataset,index):
    """Exactly the original source half of __getitem__, no target RGB/mask load."""
    from PIL import Image
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import (
        ProductObjectReplacementDataset,_seed,render_composed_background,combination_id,
        _background_prompt,_object_prompt,_joint_prompt)
    source=dataset.sources[index//12];offset=index%12
    rgb,mask=ProductObjectReplacementDataset._load_product(source,dataset.image_size)
    source_scenes=(('modern_study','cool','bright','right'),('boutique_lounge','warm','dim','left'),('concrete_loft','neutral','bright','right'))
    scene=source_scenes[_seed(source['sample_id'],dataset.split)%len(source_scenes)]
    background=render_composed_background(scene,dataset.image_size,source['sample_id'],'unified-source')
    reference=dataset._tensor(Image.composite(rgb,background,mask))
    mode='background' if offset<4 else 'object' if offset<8 else 'joint'
    _,design=dataset.by_category[source['category']][offset%4]
    target_scene=dataset.scenes[_seed(source['sample_id'],dataset.split,str(offset))%len(dataset.scenes)]
    prompt=_background_prompt(target_scene) if mode=='background' else _object_prompt(design) if mode=='object' else _joint_prompt(design,target_scene)
    return dict(reference=reference,prompt=prompt,sample_id=f"{source['sample_id']}:{mode}:{offset}",
        source_image_path=source['image_path'],source_mask_path=source['mask_path'],mode=mode,
        source_scene=combination_id(scene))


def online_embeddings(processor,embedding,prompt,device):
    from LightGenV2.tasks.t12_text_to_image.feature_cache import _qwen_prompts
    values={k:v[:,-64:].to(device) for k,v in _qwen_prompts(processor,[prompt]).items()}
    # Original producer loaded BF16 native weights, then sealed FP16 embeddings.
    # Retain that original quantization before original FP32 small-editor input.
    result=embedding(values['input_ids']).half().float()
    return result,values['attention_mask'].bool(),values


def patch(model,meter):
    from measure_salicon import one_index,shared
    for name in ('input_projection','norm','condition_projection'):meter.wrap(getattr(model.text,name),'forward','language.'+name)
    for i,layer in enumerate(model.text.layers):meter.wrap(layer,'forward',f'language.parallel_residual{i+1}',family='parallel_residual',charged=False)
    spatial=model.editor.bottleneck
    meter.wrap(spatial.condition,'forward','vision.condition_projection')
    for i in (1,2):meter.wrap(getattr(spatial,'electronic'+str(i)),'forward',f'vision.parallel_residual{i}',family='parallel_residual',charged=False)
    for mode,owner in (('language',model.text),('vision',spatial)):
        b=owner.optical;c=b.core
        for i in (1,2):meter.wrap(getattr(owner,'fusion'+str(i)),'forward',mode+f'.fusion{i}',family='fusion',charged=False)
        body=shared.meter_lib.statements(c,'encode_groups');i=shared.meter_lib.assign_index(body,'projected')
        meter.patch_statements(c,'encode_groups',[(mode+'.amplitude_encoding','amplitude_encoding_nn',True,i,i)])
        body=shared.meter_lib.statements(b,'_encode_input_fields');i=shared.meter_lib.assign_index(body,'rms')
        meter.patch_statements(b,'_encode_input_fields',[(mode+'.amplitude_rms','amplitude_encoding_nn',True,i,len(body)-1)])
        meter.wrap(c,'_direct_amplitude_load',mode+'.layout',family='layout',charged=False)
        meter.wrap(c.router,'_simulate',mode+'.router_simulation',family='simulation_only',charged=False)
        meter.wrap(b,'_simulate_detector_roi',mode+'.expert_global_simulation',family='simulation_only',charged=False)
        body=shared.meter_lib.statements(c.router,'forward');i=shared.meter_lib.assign_index(body,'energy')
        end=one_index(body,lambda s:s.startswith('weights, selected, indices ='))
        meter.patch_statements(c.router,'forward',[(mode+'.router_energy_top2','router',True,i,end)])
        meter.wrap(b.ccd_normalizer,'forward',mode+'.ccd_nonlinearity',family='ccd_readout')
        for name,head,adapter in (('expert',b.expert_readout,b.expert_output_adapter),('global',c.readout,c.output_adapter)):
            body=shared.meter_lib.statements(head,'forward_intensity')
            first=one_index(body,lambda s:s.startswith('detector_intensity =') and '.clamp_min(' in s)
            meter.patch_statements(head,'forward_intensity',[(f'{mode}.{name}_ccd_readout','ccd_readout',True,first,len(body)-1)])
            meter.wrap(adapter,'forward',f'{mode}.{name}_output_adapter',family='ccd_readout')
    for name in ('stem','down1','down2','down3','up3','up2','up1','to_delta','source_gate'):
        module=getattr(model.editor,name,None)
        if module is not None:meter.wrap(module,'forward','decoder.'+name,family='decoder')
    prior=getattr(spatial,'decoder_spatial_prior',None)
    if prior is not None:meter.wrap(prior,'forward','decoder.original_spatial_prior',family='decoder')


def sources():
    # The observation helper loads measure_ours through importlib, without a
    # sys.modules entry. Bind that real dependency explicitly as well.
    paths={Path(__file__),HERE/'stage_meter.py',HERE/'measure_ours.py',HERE/'measure_salicon.py'}
    for m in list(sys.modules.values()):
        value=getattr(m,'__file__',None)
        if value and Path(value).is_absolute():
            p=Path(value).resolve()
            if p.suffix=='.py' and p.is_relative_to(REPO):paths.add(p)
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation'):raise ValueError('New immutable CPU contract required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only environment required')
    args.output.mkdir(parents=True);write(args.output/'status.json',dict(status='identity',pid=os.getpid(),no_gpu=True))
    for path,pin in [(BODY,BODY_SHA),(DATA/'test.jsonl',TEST_SHA),(MODEL/'model.safetensors',WEIGHT_SHA),
            *[(ASSETS/'datasets'/name,pin) for name,pin in PINS.items()]]:
        if sha(path)!=pin:raise ValueError('Original asset SHA differs: '+str(path))
    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    from LightGenV2.tasks.t12_text_to_image.qwen_mini_small import PromptEmbeddingLookup
    from LightGenV2.tasks.t12_text_to_image.sealed_editor import build_sealed
    from stage_meter import StageMeter
    # Resolve observation-only lazy imports BEFORE taking the immutable source
    # snapshot. No model/data/embedding call or dummy forward is added here.
    from measure_salicon import one_index,shared
    torch.set_num_threads(8);torch.manual_seed(1042);meter=StageMeter();handles=[];shapes=[];checks=[]
    try:
        write(args.output/'status.json',dict(status='loading',pid=os.getpid(),no_gpu=True))
        model=build_sealed(torch.load(BODY,map_location='cpu',weights_only=False)).eval().requires_grad_(False)
        if sum(p.numel() for p in model.parameters())!=17026642:raise ValueError('Original17M architecture differs')
        dataset=ExpandedUnifiedProductEditDataset(DATA,'test',256,ASSETS/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt')
        cache=PromptEmbeddingLookup(ASSETS/'datasets/abo_unified_expanded_qwen_embeddings_v2.pt')
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:weight=file.get_tensor('model.language_model.embed_tokens.weight')
        embedding=torch.nn.Embedding.from_pretrained(weight,freeze=True)
        for name,module in model.named_modules():
            if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                def observe(m,a,b,name=name):
                    shapes.append(dict(test_index=meter.sample_index,module=name,kind=type(m).__name__,
                        input_shape=list(a[0].shape),output_shape=list(b.shape),weight_shape=list(m.weight.shape),groups=getattr(m,'groups',None)))
                handles.append(module.register_forward_hook(observe))
        frozen=sources()
        with torch.inference_mode():
            for i in (200,2303):
                meter.close();meter.sample_index=i;original=dataset[i];live=reference_input(dataset,i)
                if not torch.equal(original['reference'],live['reference']) or original['prompt']!=live['prompt'] or original['sample_id']!=live['sample_id']:
                    raise ValueError('Real reference-only frontend differs; target must not enter inference')
                old,oldmask,_=cache.batch([original['prompt']],torch.device('cpu'))
                current,mask,inputs=online_embeddings(processor,embedding,live['prompt'],torch.device('cpu'))
                if not torch.equal(old[0,oldmask[0]].float(),current[0,mask[0]]):raise ValueError('Original native embedding quantization/tokenizer differs')
                noise=torch.randn((1,3,256,256),generator=torch.Generator().manual_seed(1042+i))
                rng=torch.random.get_rng_state()
                expected=model(original['reference'][None],old.float(),oldmask,noise)
                patch(model,meter)
                torch.random.set_rng_state(rng)
                actual=model(live['reference'][None],current,mask,noise)
                if not torch.equal(expected,actual):raise ValueError('Original cached-input/reference vs live/observed full model differs')
                count=Counter(r['stage'] for r in meter.rows if r['sample_index']==i)
                for mode in ('language','vision'):
                    if count[mode+'.router_simulation']!=1 or count[mode+'.expert_global_simulation']!=2:raise ValueError('Original six light passes not observed')
                checks.append(dict(test_index=i,sample_id=live['sample_id'],bitwise_equal=True,reference_bitwise_equal=True,
                    native_valid_embeddings_bitwise_equal=True,input_ids=inputs['input_ids'].tolist(),attention_mask=mask.tolist(),
                    output_shape=list(actual.shape),module_call_counts=dict(count),target_passed_to_model=False))
        if torch.cuda.is_initialized() or sources()!=frozen:raise ValueError('CPU-only/source identity changed')
        write(args.output/'contract.json',dict(status='passed',no_gpu=True,body_sha256=BODY_SHA,checks=checks,
            formal_full2304_reference_psnr=34.27775097739636,stage_ledger=meter.ledger,actual_NN_shapes=shapes,
            sources=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            original_cache_used_only_as_CPU_numerical_reference=True,not_a_formal_timing_or_accuracy_evaluation=True))
        write(args.output/'status.json',dict(status='complete',no_gpu=True,pid=os.getpid(),real_inputs=2))
    except BaseException as error:write(args.output/'status.json',dict(status='failed',no_gpu=True,error=repr(error)));raise
    finally:
        meter.close()
        for handle in handles:handle.remove()


if __name__=='__main__':main()
