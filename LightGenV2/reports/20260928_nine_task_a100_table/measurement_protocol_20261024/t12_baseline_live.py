"""Original matched Qwen28 editor live-input adapter and CPU equivalence proof.

No cache as inference input, no training, no vision tower/LM head, no warmup.
The CPU oracle executes original formal export AST statements unchanged.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from t12_live_frontend import TASK,ASSETS,MODEL,BASELINE,BASELINE_SHA,WEIGHT_SHA,DATA,TEST_SHA,PINS,reference_input,sha,write,sources
EXPORT='LightGenV2/tasks/t12_text_to_image/export_matched_summary.py'
EXPORT_COMMIT='c657f959d877635549d68cfbb8bba4c668273adf'
EXPORT_SHA='558e2a81c5735406fdefcea074a3b4063b5e70eb6f21ebe94fca7992b379f3e6'
DEPENDENCY_RECORD='maintenance/storage/T12_HISTORICAL_SEVEN_MODEL_IDENTITY_20261004.json'
DEPENDENCY_RECORD_SHA='4a567a4a956cf5815514d14a43edc6c9dbc42002f508e60633b85e22b158ea7b'


def dependency_records():
    raw=subprocess.check_output(['git','show','HEAD:'+DEPENDENCY_RECORD],cwd=REPO)
    if hashlib.sha256(raw).hexdigest()!=DEPENDENCY_RECORD_SHA:raise ValueError('Original registered dependency record changed')
    value=json.loads(raw);records=value['frozen_dependencies']['files']
    wanted=['bk-sdm-v2-tiny/unet/config.json','bk-sdm-v2-tiny/unet/diffusion_pytorch_model.fp16.safetensors',
        'sd-turbo-fp16/vae/config.json','sd-turbo-fp16/vae/diffusion_pytorch_model.fp16.safetensors',
        'sd-turbo-fp16/scheduler/scheduler_config.json']
    result=[]
    for name in wanted:
        matches=[r for r in records if r['path']==(ASSETS/'models'/name).as_posix()]
        if len(matches)!=1:raise ValueError('Missing/ambiguous original dependency identity')
        result.append(dict(relative_path='models/'+name,bytes=matches[0]['bytes'],sha256=matches[0]['sha256']))
    return result


def bind_dependencies():
    records=dependency_records()
    for item in records:
        path=ASSETS/item['relative_path']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Original frozen VAE/UNet construction/scheduler asset differs: '+str(path))
    return dict(registered_identity_record=DEPENDENCY_RECORD,registered_identity_record_sha256=DEPENDENCY_RECORD_SHA,
        fresh_assets=records,scope='Original formal-loader paths bound to registered complete-file hashes; old benchmark argv/open-history not proved')


def original_statements():
    raw=subprocess.check_output(['git','show',EXPORT_COMMIT+':'+EXPORT],cwd=REPO)
    if hashlib.sha256(raw).hexdigest()!=EXPORT_SHA:raise ValueError('Original formal export changed')
    tree=ast.parse(raw);result=[]
    for name in ('hidden','valid','condition','pooled','latent','edited','pred'):
        matches=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
            target=ast.unparse(node.targets[0]);value=ast.unparse(node.value)
            if name=='condition':ok=target=='conditions[prompt]' and value.startswith('(hidden * valid).sum(')
            else:
                ok=target==name
                if name=='hidden':ok=ok and value.startswith('qwen(')
                if name=='pred':ok=ok and value.startswith('vae.decode(')
            if ok:matches.append(node)
        if len(matches)!=1:raise ValueError('Ambiguous original formal statement: '+name)
        result.append(matches[0])
    return result


def load_runtime(device):
    import torch
    from diffusers import AutoencoderKL,EulerDiscreteScheduler,UNet2DConditionModel
    from LightGenV2.tasks.t12_text_to_image.half_qwen import load_half_qwen_text_encoder
    from LightGenV2.tasks.t12_text_to_image.electronic_turbo_infer import _load_adapter
    from LightGenV2.tasks.t12_text_to_image.product_repair_model import expand_reference_conditioning
    dependencies=bind_dependencies()
    if sha(BASELINE)!=BASELINE_SHA or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:
        raise ValueError('Adopted matched head/native language weights differ')
    payload=torch.load(BASELINE,map_location='cpu',weights_only=False,mmap=True)
    if not payload.get('matched_training',{}).get('step',0):raise ValueError('Original matched trained checkpoint required')
    qwen,processor,report=load_half_qwen_text_encoder(MODEL,device,keep_layers=28)
    adapter,_=_load_adapter(ASSETS/'runs/677deac6/qwen_sd_turbo_one_step_text_aug_seed42/best_adapter.pt',device)
    adapter.load_state_dict(payload['adapter'],strict=True)
    unet=UNet2DConditionModel.from_pretrained(ASSETS/'models/bk-sdm-v2-tiny',subfolder='unet',variant='fp16',
        torch_dtype=torch.float32,local_files_only=True)
    expand_reference_conditioning(unet);unet.load_state_dict(payload['unet'],strict=True);unet=unet.to(device)
    vae=AutoencoderKL.from_pretrained(ASSETS/'models/sd-turbo-fp16',subfolder='vae',variant='fp16',
        torch_dtype=torch.float16,local_files_only=True).to(device)
    scheduler=EulerDiscreteScheduler.from_pretrained(ASSETS/'models/sd-turbo-fp16',subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(1,device=device)
    for module in (qwen,adapter,unet,vae):module.eval().requires_grad_(False)
    if report['language_layers_retained']!=28 or report['vision_tower_used'] or report['language_model_head_used']:
        raise ValueError('Original full text-only 28-layer architecture required')
    return SimpleNamespace(qwen=qwen,processor=processor,adapter=adapter,unet=unet,vae=vae,
        scheduler=scheduler,payload=payload,device=device,qwen_report=report,frozen_dependency_binding=dependencies)


def inputs(runtime,prompt):
    from LightGenV2.tasks.t12_text_to_image.feature_cache import _qwen_prompts
    return {k:v[:,-64:].to(runtime.device) for k,v in _qwen_prompts(runtime.processor,[prompt]).items()}


def forward(runtime,reference,tokens,index):
    """Literal original formal arithmetic, no condition dictionary reuse."""
    import torch
    from LightGenV2.tasks.t12_text_to_image.product_repair_model import one_step_edit
    from LightGenV2.tasks.t12_text_to_image.export_matched_summary import noise_batch
    with torch.autocast(runtime.device.type,dtype=torch.float16):
        hidden=runtime.qwen(**tokens,use_cache=False,return_dict=True).last_hidden_state.float()
        valid=tokens['attention_mask'].to(hidden.dtype).unsqueeze(-1)
        pooled=(hidden*valid).sum(1)/valid.sum(1).clamp_min(1)
        latent=runtime.vae.encode(reference.half()).latent_dist.mode()*runtime.vae.config.scaling_factor
        edited=one_step_edit(runtime.unet,noise_batch(len(reference),runtime.device,index,1042),latent.float(),
            runtime.adapter.condition(pooled.float()),runtime.scheduler.sigmas[0],
            residual_scale=float(runtime.payload['training_config']['residual_scale']),
            noise_scale=float(runtime.payload['training_config']['noise_scale']))
        return runtime.vae.decode(edited.half()/runtime.vae.config.scaling_factor,return_dict=False)[0].float()


def gold_forward(runtime,reference,tokens,prompt,index):
    import torch
    from LightGenV2.tasks.t12_text_to_image.export_matched_summary import noise_batch
    from LightGenV2.tasks.t12_text_to_image.product_repair_model import one_step_edit
    scope=dict(torch=torch,qwen=runtime.qwen,vae=runtime.vae,unet=runtime.unet,adapter=runtime.adapter,
        scheduler=runtime.scheduler,payload=runtime.payload,tokens=tokens,ref=reference,conditions={},
        prompt=prompt,batch={'prompt':[prompt]},device=runtime.device,start=index,args=SimpleNamespace(seed=1042),
        noise_batch=noise_batch,one_step_edit=one_step_edit)
    # The only platform substitution is torch.autocast(cpu) instead of cuda;
    # original load_half_qwen_text_encoder also defines its CPU weights as FP32.
    code=compile(ast.fix_missing_locations(ast.Module(body=original_statements(),type_ignores=[])),EXPORT,'exec')
    with torch.autocast(runtime.device.type,dtype=torch.float16):exec(code,scope)
    return scope['pred']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable canonical CPU proof only')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only environment required')
    args.output.mkdir(parents=True);write(args.output/'status.json',dict(status='identity',pid=os.getpid(),no_gpu=True))
    original_statements()
    for path,pin in [(DATA/'test.jsonl',TEST_SHA),*[(ASSETS/'datasets'/name,pin) for name,pin in PINS.items()]]:
        if sha(path)!=pin:raise ValueError('Original input identity differs')
    import torch
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    # Load the original oracle module before freezing the source inventory;
    # neither importing it nor reading its AST performs model inference.
    from LightGenV2.tasks.t12_text_to_image.export_matched_summary import noise_batch
    torch.set_num_threads(8);checks=[];shapes=[];handles=[]
    try:
        write(args.output/'status.json',dict(status='loading',pid=os.getpid(),no_gpu=True))
        runtime=load_runtime(torch.device('cpu'))
        dataset=ExpandedUnifiedProductEditDataset(DATA,'test',256,ASSETS/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt')
        frozen=sources()
        with torch.inference_mode():
            for index in (200,2303):
                original=dataset[index];live=reference_input(dataset,index)
                if (not torch.equal(original['reference'],live['reference']) or original['prompt']!=live['prompt']
                        or original['sample_id']!=live['sample_id']):raise ValueError('Reference-only original input differs')
                tokens=inputs(runtime,live['prompt'])
                expected=gold_forward(runtime,original['reference'][None],tokens,live['prompt'],index)
                for group in ('qwen','adapter','unet','vae'):
                    for name,module in getattr(runtime,group).named_modules():
                        if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                            def observe(m,a,b,name=group+'.'+name):
                                shapes.append(dict(test_index=index,module=name,kind=type(m).__name__,
                                    input_shape=list(a[0].shape),output_shape=list(b.shape),
                                    weight_shape=list(m.weight.shape),groups=getattr(m,'groups',None)))
                            handles.append(module.register_forward_hook(observe))
                actual=forward(runtime,live['reference'][None],tokens,index)
                for handle in handles:handle.remove()
                handles=[]
                if not torch.equal(expected,actual):raise ValueError('Original formal AST and live/observed full editor differ')
                checks.append(dict(test_index=index,sample_id=live['sample_id'],bitwise_equal=True,
                    original_reference_bitwise_equal=True,target_passed_to_model=False,
                    input_ids=tokens['input_ids'].tolist(),attention_mask=tokens['attention_mask'].tolist(),
                    output_shape=list(actual.shape)))
        if torch.cuda.is_initialized() or frozen!=sources():raise ValueError('CPU-only/source contract changed')
        write(args.output/'contract.json',dict(status='passed',no_gpu=True,checks=checks,
            baseline_sha256=BASELINE_SHA,formal_full2304_reference_psnr=27.26045327136914,
            original_export_source_commit=EXPORT_COMMIT,original_export_source_sha256=EXPORT_SHA,
            qwen_report=runtime.qwen_report,frozen_dependency_binding=runtime.frozen_dependency_binding,
            actual_NN_shapes=shapes,sources=frozen,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            CPU_platform='original CPU FP32 language weights; original FP16 VAE; identical CPU FP16 autocast for oracle and adapter',
            cached_conditions_used_as_online_input=False,not_formal_timing_or_full_test_evaluation=True))
        write(args.output/'status.json',dict(status='complete',no_gpu=True,real_inputs=2,pid=os.getpid()))
    except BaseException as error:
        write(args.output/'status.json',dict(status='failed',no_gpu=True,error=repr(error)));raise
    finally:
        for handle in handles:handle.remove()


if __name__=='__main__':main()
