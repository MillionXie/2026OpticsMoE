"""Raw paired-model DEV generation; never opens pairs.json/target latent cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import time
import statistics
import torch
from PIL import Image
from .qwen3_scene_conditioning import QwenSDXLConditions
from .qwen3_scene_paired_train import new_qwen, raw_scene_features, pipeline
from .white_reference_scene_probe import GPU, digest, write, checked_tokens


def inference_inputs(data,scope='dev'):
    if scope not in ['dev','test']:raise ValueError('Explicit DEV or frozen TEST scope required')
    inputs = json.loads((data/('inference_inputs.json' if scope=='test' else 'inference_dev_inputs.json')).read_text())
    safe = {'sample_id','reference','prompt','seed','reference_id','reference_sha256','category',
            'condition_id','native_caption','reward_caption','prompt_profile'}
    if not inputs or not all(set(row) == safe for row in inputs):
        raise ValueError('Only reference/text/seed and public request metadata allowed at inference')
    return inputs


def frozen_test_protocol(a,state,previous):
    if a.frozen_evaluation is None or not a.frozen_model_key:raise ValueError('TEST requires a previously frozen model/protocol manifest')
    frozen=json.loads(a.frozen_evaluation.read_text());report=json.loads((a.data/'report.json').read_text())
    assert frozen['schema']=='scene_frozen_evaluation_v1' and frozen['development_selection_finished']
    assert not frozen['TEST_used_for_model_selection'] and report['complete'] and report['role_disjoint']
    assert frozen['frozen_before_first_TEST_prediction']
    assert frozen['prompt_profile']==report['prompt_profile']=='two_sentence_furnished_room_v3'
    assert frozen['request_sha256']==report['request_sha256']==digest(a.data/'inference_inputs.json')
    assert report['current_pairs_sha256']==state['pair_sha256']
    model=frozen['models'][a.frozen_model_key]
    assert model['checkpoint_sha256']==previous['best_sha256'] and model['sampling_steps']==a.steps
    assert model['IP_adapter_scale']==a.ip_scale and model['resolution']==state['resolution']==512
    assert model['Qwen_input_side']==state['Qwen_input_side']==256 and model['variant']==state['variant']
    assert model['joint_image_context']==state.get('joint_image_context',False)
    assert model['sampling_contract']==state.get('sampling_contract')
    assert model.get('task_LoRA_fused',False)==bool(getattr(a,'fuse_task_lora',False))
    assert model['optical_frontend_sha256']==(state.get('optical_frontend') or {}).get('checkpoint_sha256')
    fast=getattr(a,'lcm_adapter',None)
    if fast:
        from .scene_fast_tail import tail_configuration
        assert model['fast_tail']==tail_configuration(fast,getattr(a,'lcm_guidance',1.))
    else:assert model.get('fast_tail') is None
    if a.measure_software_latency:raise ValueError('Use the separate fixed DEV timing protocol; TEST500 is generation-only')
    return {'manifest_sha256':digest(a.frozen_evaluation),'model_key':a.frozen_model_key,'descriptor':model}


def install_optical_frontend(qwen,folder,features,baseline_cache):
    from .qwen_vl_postfree import apply_postfree
    from .qwen_vl_hybrid import load_macro_state
    if features is None:raise ValueError('LG teacher feature provenance required')
    report=json.loads((folder/'report.json').read_text())
    assert report['complete'] and report['stage']=='distill' and report['steps']>0
    assert digest(folder/'best_checkpoint.pt')==report['best_sha256']
    state=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['schema']=='qwen3_scene_optical_frontend_v1' and state['step']>0
    feature_report=json.loads((features/'report.json').read_text())
    assert state['feature_cache_sha256']==feature_report['cache_sha256']
    assert json.loads((features/'model_assets.json').read_text())==json.loads((baseline_cache/'model_assets.json').read_text())['Qwen']
    vision_class=type(qwen.model.visual.blocks[0]);language_class=type(qwen.model.language_model.layers[0])
    macros=apply_postfree(qwen,vision_pairs=1,language_pairs=1,residual_width=state['config']['residual_width'])
    load_macro_state(macros,state['macros']);qwen.eval().requires_grad_(False)
    counts={'vision':sum(type(x)==vision_class for x in qwen.model.visual.blocks),
            'language':sum(type(x)==language_class for x in qwen.model.language_model.layers)}
    assert counts=={'vision':6,'language':4}
    assert all(.6<=f.alpha.item()<=.900001 for m in macros.values() for f in m.fusions)
    return {'checkpoint_sha256':report['best_sha256'],'retained_original_TF':counts,'actual_removed_TF':42,
            'feature_distillation_only_generator_recovery_not_yet_trained':True}


def render(a):
    from peft import LoraConfig
    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    state = torch.load(a.checkpoint/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    a.qwen_side=state['Qwen_input_side'];resolution=state['resolution']
    previous = json.loads((a.checkpoint/'report.json').read_text())
    assert previous['complete'] and digest(a.checkpoint/'best_checkpoint.pt') == previous['best_sha256']
    sampling=state.get('sampling_contract')
    fast=None
    fuse_task=bool(getattr(a,'fuse_task_lora',False))
    if getattr(a,'lcm_adapter',None):
        from .scene_fast_tail import tail_configuration
        fast=tail_configuration(a.lcm_adapter,getattr(a,'lcm_guidance',1.))
        assert sampling is None and a.steps in [4,8],'Do not stack an unvalidated task-distilled tail with pretrained LCM'
    elif getattr(a,'lcm_guidance',1.)!=1.:raise ValueError('LCM guidance requires a checked LCM tail')
    if sampling:
        assert sampling['kind'] in ['guided_Euler12_to_conditional6_v1','progressive_guided_Euler_v2'] and a.steps==sampling['steps']
        assert len(sampling['timesteps'])==a.steps and all(x>y for x,y in zip(sampling['timesteps'],sampling['timesteps'][1:]))
        assert a.ip_scale==sampling['IP_scale'] and sampling['CFG']==1.
    elif not fast and a.steps in [4,6,7,8,15]:raise ValueError('This few-step budget requires its verified trained/distilled tail')
    is_test=getattr(a,'scope','dev')=='test'
    frozen=frozen_test_protocol(a,state,previous) if is_test else None
    inputs = inference_inputs(a.data,'test' if is_test else 'dev')
    if is_test:
        assert len(inputs)==500 and len({r['sample_id'] for r in inputs})==500
        assets=json.loads((a.cache/'model_assets.json').read_text())
        from .qwen3_scene_conditioning import qwen_asset_manifest
        from .white_reference_scene_probe import model_manifest
        assert assets['SDXL']==model_manifest(a.model_path) and assets['IPAdapter']==model_manifest(a.adapter_path)
        if state['variant']=='qwen':assert assets['Qwen']==qwen_asset_manifest(a.qwen)
    rows = inputs if is_test else [r for r in inputs if r['condition_id'] in ['0','1']]
    expected=None;latencies=[]
    if a.measure_software_latency:
        if a.reference_run is None:raise ValueError('Timing requires a completed same-weight quality run for output identity')
        verified=json.loads((a.reference_run/'report.json').read_text())
        assert verified['complete'] and verified['checkpoint_sha256']==previous['best_sha256'] and verified['sampling_steps']==a.steps
        assert verified.get('IP_adapter_scale',.25)==a.ip_scale,'Timing and quality reference strengths differ'
        assert verified.get('fast_tail')==fast,'Timing and quality tail identities differ'
        assert verified.get('task_LoRA_fused',False)==fuse_task,'Timing and quality task fusion graphs differ'
        expected={r['sample_id']:r for r in json.loads((a.reference_run/'images.json').read_text())}
        rows=[r for category in ['lamp','pillow'] for r in [x for x in rows if x['category']==category][:4]]
        assert len(rows)==8
    qwen = processor = head = identity = None
    optical=None
    exact_identity=False
    if state.get('optical_frontend') and a.lg_frontend is None:raise ValueError('LG-trained head requires its bound optical frontend')
    if state['variant'] == 'qwen':
        qwen, processor = new_qwen(a)
        head = QwenSDXLConditions(joint_image_context=state.get('joint_image_context',False)).cuda().eval().requires_grad_(False)
        head.load_state_dict(state['head'], strict=True)
        pipe = pipeline(a)
        if not is_test:
            identity = torch.load(a.cache/'DEV_frontend_identity_only.pt', map_location='cpu', weights_only=False)
            assert not identity['contains_targets_or_masks']
        if a.lg_frontend:optical=install_optical_frontend(qwen,a.lg_frontend,a.lg_features,a.cache)
        if state.get('optical_frontend'):
            assert optical['checkpoint_sha256']==state['optical_frontend']['checkpoint_sha256']
            optical['feature_distillation_only_generator_recovery_not_yet_trained']=False
        exact_identity=not is_test and (optical is None or identity.get('optical_frontend',{}).get('checkpoint_sha256')==optical['checkpoint_sha256'])
    else:
        if a.lg_frontend:raise ValueError('LG replacement belongs to Qwen branch only')
        encoder = CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',
            torch_dtype=torch.bfloat16, local_files_only=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, variant='fp16',
            torch_dtype=torch.bfloat16, image_encoder=encoder, local_files_only=True).to('cuda')
        pipe.load_ip_adapter(str(a.adapter_path), subfolder='sdxl_models',
            weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder=None, local_files_only=True)
        pipe.set_ip_adapter_scale(.25); pipe.set_progress_bar_config(disable=True)
    if fast:
        from .scene_fast_tail import install_tail,checked_tail
        assert checked_tail(a.lcm_adapter)==install_tail(pipe,a.lcm_adapter)
    base_parameter_shapes={n:tuple(p.shape) for n,p in pipe.unet.named_parameters()}
    pipe.unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian',
        target_modules=['to_q','to_k','to_v','to_out.0']), adapter_name='scene_pair')
    named = dict(pipe.unet.named_parameters())
    assert set(state['unet_lora']) == {n for n in named if 'lora_' in n}
    with torch.no_grad():
        for n, value in state['unet_lora'].items(): named[n].copy_(value)
    task_fusion=None
    if fuse_task:
        from .scene_fast_tail import fuse_task_adapter
        task_fusion=fuse_task_adapter(pipe,base_parameter_shapes)
    # Expose reference strength as an explicit DEV ablation. The historical
    # default is unchanged; strength never adds another RGB encoder or network.
    pipe.set_ip_adapter_scale(a.ip_scale)
    for c in pipe.components.values():
        if isinstance(c,torch.nn.Module): c.eval().requires_grad_(False)
    records = []
    for i,row in enumerate(rows):
        src = a.data/row['reference']; assert digest(src) == row['reference_sha256']
        dst = a.output/(row['reference_id']+'.png')
        if not dst.exists(): shutil.copyfile(src,dst)
        with Image.open(src) as im: reference = im.convert('RGB').copy()
        if a.measure_software_latency:torch.cuda.synchronize();began=time.perf_counter()
        generator = torch.Generator(device='cuda').manual_seed(row['seed'])
        noise = torch.randn((1,4,resolution//8,resolution//8), generator=generator, device='cuda', dtype=torch.bfloat16)
        kwargs = dict(width=resolution,height=resolution,num_inference_steps=a.steps,guidance_scale=fast['CFG'] if fast else 7.5,latents=noise,generator=generator)
        if sampling:
            kwargs.pop('num_inference_steps')
            kwargs.update(timesteps=sampling['timesteps'],guidance_scale=sampling['CFG'])
        conditioning_side=int(state.get('SDXL_conditioning_size',resolution))
        kwargs.update(original_size=(conditioning_side,conditioning_side),target_size=(conditioning_side,conditioning_side))
        with torch.inference_mode():
            if head is not None:
                h,v,ids = raw_scene_features(qwen,processor,reference,row['prompt'],a.qwen_side,keep_on_device=a.measure_software_latency)
                cond = head(h.unsqueeze(0).cuda().float(),v.unsqueeze(0).cuda().float())
                image_conditions=cond['image'].bfloat16()
                if kwargs['guidance_scale']>1:
                    image_conditions=torch.cat([state['negative_image'].cuda().bfloat16(),image_conditions],dim=0)
                images=image_conditions.unsqueeze(1)
                kwargs.update(prompt_embeds=cond['text'].bfloat16(),pooled_prompt_embeds=cond['pooled'].bfloat16(),
                    negative_prompt_embeds=state['negative_text'].cuda().bfloat16(),
                    negative_pooled_prompt_embeds=state['negative_pooled'].cuda().bfloat16(),ip_adapter_image_embeds=[images])
            else:
                for t in [pipe.tokenizer,pipe.tokenizer_2]: checked_tokens(t,row['native_caption'])
                kwargs.update(prompt=row['native_caption'],negative_prompt=state['negative_caption'],ip_adapter_image=reference)
            image = pipe(**kwargs).images[0]
        if a.measure_software_latency:
            torch.cuda.synchronize();latencies.append((time.perf_counter()-began)*1000)
        # Diagnostics and disk output are outside the RGB/text-ready -> CPU RGB
        # window. The first real generation remains in the measured requests.
        noise_sha = hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
        if head is not None and identity is not None:
            key = hashlib.sha256(src.read_bytes()+b'\0'+row['prompt'].encode()+f'\0Qwen{a.qwen_side}'.encode()).hexdigest();old=identity['features'][key]
            assert torch.equal(ids.cpu(),old['input_ids'])
            if exact_identity:assert torch.equal(h.cpu(),old['hidden']) and torch.equal(v.cpu(),old['visual'])
            else:assert h.shape==old['hidden'].shape and v.shape==old['visual'].shape and torch.isfinite(h).all() and torch.isfinite(v).all()
        elif head is not None:
            assert h.shape==(ids.shape[1],2048) and v.shape==((a.qwen_side//32)**2,2048)
            assert torch.isfinite(h).all() and torch.isfinite(v).all()
        name = row['sample_id']+'.png'; image.save(a.output/name)
        if expected is not None:
            wanted=expected[row['sample_id']]
            assert digest(a.output/name)==wanted['generated_sha256'],'Timing graph changed output PNG'
            assert noise_sha==wanted['initial_latent_sha256']
        routes=None
        if optical is not None:
            routes={m:x.diagnostics() for m,x in qwen.model.hybrid.items()}
        records.append({**row,'split':'test' if is_test else 'dev','generated_PNG':name,'generated_sha256':digest(a.output/name),
            'input_sha256':digest(src),'initial_latent_sha256':noise_sha,'model':state['variant'],
            'raw_Qwen_identity_exact':exact_identity,'resolution':resolution,'sampling_steps':a.steps,
            'IP_adapter_scale':a.ip_scale,
            'sampling_contract':sampling,
            'fast_tail':fast,
            'task_LoRA_fused':fuse_task,
            'actual_raw_optical_routing':routes,
            'optical_frontend':optical})
        write(a.output/'images.json',records)
        write(a.output/'progress.json',{'stage':'frozen_raw_TEST_render' if is_test else 'independent_raw_DEV_render','done':i+1,'total':len(rows)})
    generator_count = sum(p.numel() for p in {id(p):p for c in pipe.components.values()
        if isinstance(c,torch.nn.Module) for p in c.parameters()}.values())
    if latencies:
        write(a.output/'software_latency.json',{'boundary':'RGB PIL and text ready -> generated RGB returned to CPU; no disk output or identity checks in timed window',
            'n':len(latencies),'requests_ms':latencies,'mean_ms':statistics.mean(latencies),'median_ms':statistics.median(latencies),
            'first_request_ms':latencies[0],'model_load_excluded':True,'extra_warmup':False,'batch_size':1,
            'output_PNG_identity_vs_completed_quality_run':True,'sampling_steps':a.steps,
            'IP_adapter_scale':a.ip_scale,
            'sampling_contract':sampling,
            'fast_tail':fast,
            'contains_GPU_FFT_optical_simulation':optical is not None,'includes_electronic_residual':True,
            'physical_hardware_estimate_or_measurement':False,'single_process_round_only':True,
            'LoRA_unmerged_for_both_models':not fuse_task,'task_LoRA_fused':fuse_task,'task_fusion':task_fusion})
    write(a.output/'report.json',{'complete':True,'scope':'TEST' if is_test else ('DEV_software_latency' if latencies else 'DEV'),'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'generated_count':len(records),'prompt_profile':rows[0]['prompt_profile'],'training_performed':False,
        'checkpoint_sha256':previous['best_sha256'],'paired_diffusion_training_performed':True,
        'independent_TEST_opened':is_test,'GT_mask_or_target_latent_files_opened':False,
        'raw_Qwen_identity_checked':exact_identity,'image_feedback':False,'quality_accepted':False,
        'optical_frontend':optical,'raw_input_ids_match_teacher':head is not None and identity is not None,
        'CLIP_encoders_loaded_at_inference':head is None,'precision':'BF16','resolution':resolution,'sampling_steps':a.steps,
        'IP_adapter_scale':a.ip_scale,
        'sampling_contract':sampling,
        'fast_tail':fast,
        'task_LoRA_fused':fuse_task,'task_fusion':task_fusion,
        'joint_image_context':state.get('joint_image_context',False),'frozen_TEST_protocol':frozen,
        'DEV_or_TRAIN_feature_tensors_opened':not is_test and head is not None,
        'generator_loaded_parameters':generator_count,'Qwen_parameters':0 if qwen is None else sum(p.numel() for p in qwen.parameters()),
        'head_parameters':0 if head is None else sum(p.numel() for p in head.parameters()),
        'factual_evaluation_captions':True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data','checkpoint','qwen','cache','model-path','adapter-path','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--lg-frontend',type=Path);p.add_argument('--lg-features',type=Path)
    p.add_argument('--lcm-adapter',type=Path,help='Pinned, SHA-verified official SDXL LCM-LoRA; requires 4 or 8 steps, no extra input encoder')
    p.add_argument('--fuse-task-lora',action='store_true',help='Fair optional merge for either model; requires a new quality run before timing')
    p.add_argument('--lcm-guidance',type=float,choices=[1.,1.5,2.],default=1.,help='Explicit low LCM guidance; >1 includes real classifier-free double batch cost')
    p.add_argument('--steps',type=int,choices=[4,6,7,8,12,15,30],default=30,help='4/8 require verified LCM; 6/7/15 require matched learned tail; 12/30 retain original sampler')
    p.add_argument('--ip-scale',type=float,default=.25,help='Explicit reference-conditioning strength, historical default .25')
    p.add_argument('--measure-software-latency',action='store_true');p.add_argument('--reference-run',type=Path)
    p.add_argument('--scope',choices=['dev','test'],default='dev')
    p.add_argument('--frozen-evaluation',type=Path);p.add_argument('--frozen-model-key')
    a = p.parse_args()
    if not 0<=a.ip_scale<=1:raise ValueError('Reference scale must be in [0,1]')
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT','')) != 40 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Fresh run, published source and GPU6 required')
    a.output.mkdir(parents=True); torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ['T12_SOURCE_COMMIT'],
                                  'source_sha256':digest(Path(__file__))})
    try: render(a)
    except Exception:
        write(a.output/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__': main()
