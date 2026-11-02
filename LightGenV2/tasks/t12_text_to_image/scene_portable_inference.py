"""Portable raw-RGB/text LG2 full70/eight-step simulation, without camera SDKs.

Reuses the published renderer, F6 installation, head, LoRA fusion, depth graph,
sampling grid and original VAE. Low-memory execution changes residency only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

GENERATOR='aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8'
FRONTEND='6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e'
PAIRS='c2e30cbdbd871094df8827c78f4a02bc23305a0e39898ab53c6acf4178835fd0'
REQUESTS='f132238c3a59efb1308649909c96ce3669e02c7420d598bb29183536720f9f47'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while True:
            block=stream.read(8*1024*1024)
            if not block:break
            digest.update(block)
    return digest.hexdigest()


def inside(root,relative):
    value=Path(relative)
    if value.is_absolute()or'..'in value.parts:raise ValueError('Unsafe bundle path')
    result=(root/value).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError('Asset escaped bundle')
    return result


def checked_manifest(value):
    if (value.get('schema')!='t12_portable_scene_assets_v1'or value.get('complete')is not True
        or value.get('generator_sha256')!=GENERATOR or value.get('frontend_sha256')!=FRONTEND
        or value.get('pairs_sha256')!=PAIRS or value.get('requests_sha256')!=REQUESTS
        or(value.get('TRAIN'),value.get('TEST'),value.get('UNet_blocks'),value.get('steps'))!=(624,500,70,8)
        or value.get('cached_Qwen_features_included')is not False):
        raise ValueError('Require the approved aa2/F6/full70/eight-step asset set')
    return value


def subset(rows,metadata,limit):
    if type(limit)is not int or not 1<=limit<=500:raise ValueError('Explicit one-to500 requests required')
    if len(rows)!=500:raise ValueError('Validate the complete canonical cohort before subsetting')
    return rows[:limit],dict(metadata,portable_subset_count=limit,complete_original_cohort=500)


def verify(root):
    version=json.loads((root/'SOURCE_VERSION.json').read_text())
    if len(version.get('source_commit',''))!=40:raise ValueError('Published Git source identity required')
    for relative,item in version['files'].items():
        if sha(inside(root.parent,relative))!=item['sha256']:
            raise ValueError('Published source changed: '+relative)
    value=checked_manifest(json.loads((root/'DATA_manifest.json').read_text()))
    for relative,item in value['files'].items():
        path=inside(root,relative)
        if path.stat().st_size!=item['bytes']or sha(path)!=item['sha256']:
            raise ValueError('Bundle asset identity drift: '+relative)
    checks={'weights/LG2/best_checkpoint.pt':GENERATOR,'weights/F6/best_checkpoint.pt':FRONTEND,
        'data/scene_expanded_reviewed_TRAIN_TEST500_20261101/pairs.json':PAIRS,
        'data/scene_expanded_reviewed_TRAIN_TEST500_20261101/inference_inputs.json':REQUESTS}
    for relative,digest in checks.items():
        if sha(inside(root,relative))!=digest:raise ValueError('Wrong approved weight or data')
    return value


def infer(a):
    # Establish the per-process device before importing any CUDA-using modules.
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    devices={int(x.split(',')[0]):x.split(',')[1:]for x in lines}
    if a.device not in devices:raise ValueError('Requested CUDA device is unavailable')
    uuid,total=[x.strip()for x in devices[a.device]]
    occupied=[x for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()if x.startswith(uuid)]
    if occupied:raise ValueError('Selected GPU is occupied; preserve other jobs')
    os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['T12_GPU_UUID']=uuid
    version=json.loads((a.bundle/'SOURCE_VERSION.json').read_text())
    os.environ['T12_SOURCE_COMMIT']=version['source_commit']
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    mode=a.offload
    if mode=='auto':mode='sequential'if int(total)<18000 else'resident'
    import torch
    from . import scene_paired_render as renderer
    from . import scene_compact_candidates as compact
    from .scene_instruction_middle import main as run
    from .scene_expanded_middle import checked_profile,checked_training
    from .scene_expanded_selection_render import checked_revision,public_inputs,public_report
    from .scene_lab_middle8 import checked_state
    binding=json.loads((a.bundle/'provenance/approved_handoff/binding.json').read_text())
    checkpoint=a.bundle/'weights/LG2'
    checked_state(torch.load(checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False),
        json.loads((checkpoint/'report.json').read_text()),binding)
    original_pipeline=renderer.pipeline;original_inputs=renderer.inference_inputs;original_observer=compact.GraphObserver
    if mode!='resident':
        def cpu_pipeline(args):
            from diffusers import StableDiffusionXLPipeline
            from .scene_precision import dtype
            pipe=StableDiffusionXLPipeline.from_pretrained(args.model_path,variant='fp16',torch_dtype=dtype(args),
                text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,image_encoder=None,local_files_only=True)
            pipe.vae.to(dtype=torch.bfloat16)
            pipe.load_ip_adapter(str(args.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
            pipe.set_ip_adapter_scale(.25);pipe.set_progress_bar_config(disable=True)
            for component in pipe.components.values():
                if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
            return pipe
        renderer.pipeline=cpu_pipeline
        class OffloadObserver(original_observer):
            def prepare(self,qwen,processor,head,pipe):
                result=super().prepare(qwen,processor,head,pipe)
                (pipe.enable_sequential_cpu_offload if mode=='sequential'else pipe.enable_model_cpu_offload)(gpu_id=0)
                return result
        compact.GraphObserver=OffloadObserver
    def revision(*args):
        rows,metadata=checked_revision(*args)
        return (rows,metadata)if a.limit==500 else subset(rows,metadata,a.limit)
    data=a.bundle/'data/scene_expanded_reviewed_TRAIN_TEST500_20261101'
    sys.argv=[sys.argv[0],'--middle-mode','render','--middle-profile',str(a.bundle/'profiles/full_UNet_profile.json'),
        '--data',str(data),'--prompt-revision',str(data),'--checkpoint',str(checkpoint),
        '--cache',str(a.bundle/'provenance/bound_conditions'),'--qwen',str(a.bundle/'models/Qwen'),
        '--model-path',str(a.bundle/'models/SDXL'),'--adapter-path',str(a.bundle/'models/IPAdapter'),
        '--steps','8','--fuse-task-lora','--lg-frontend',str(a.bundle/'weights/F6'),
        '--lg-features',str(a.bundle/'provenance/frontend_features'),'--output',str(a.output)]
    renderer.inference_inputs=public_inputs
    try:
        run(checked_profile,checked_training,revision,expected_contexts=a.limit)
        if a.limit==500:public_report(a.output)
        else:
            rows=json.loads((a.output/'images.json').read_text())
            for row in rows:row['split']='test'
            (a.output/'images.json').write_text(json.dumps(rows,indent=2))
        report=json.loads((a.output/'report.json').read_text())
        if a.limit!=500:report.update(scope='portable_TEST_smoke_subset',full500_scored=False,TEST_used_for_model_selection=False)
        report.update(portable_simulation=True,physical_hardware_used=False,residency=mode,
            source_bundle_commit=version['source_commit'],raw_RGB_and_full_text_recomputed=True,
            training_feature_cache_used=False,new_speed_claim=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        (a.output/'portable_execution.json').write_text(json.dumps(dict(GPU_UUID=uuid,GPU_name=torch.cuda.get_device_name(0),
            memory_total_MiB=int(total),residency=mode,source_commit=version['source_commit'],entry_sha256=sha(__file__)),indent=2))
    finally:
        renderer.pipeline=original_pipeline;renderer.inference_inputs=original_inputs;compact.GraphObserver=original_observer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--verify-only',action='store_true')
    p.add_argument('--device',type=int,default=0);p.add_argument('--limit',type=int,default=500)
    p.add_argument('--offload',choices=['auto','resident','model','sequential'],default='auto')
    p.add_argument('--output',type=Path);a=p.parse_args();a.bundle=a.bundle.resolve()
    if not 1<=a.limit<=500:p.error('--limit must be between1 and500')
    verify(a.bundle)
    if a.verify_only:
        print(json.dumps(dict(complete=True,all_assets_SHA_verified=True,GPU_initialized=False)));return
    if a.output is None or a.output.exists():raise ValueError('Choose a fresh inference output directory')
    infer(a)


if __name__=='__main__':main()
