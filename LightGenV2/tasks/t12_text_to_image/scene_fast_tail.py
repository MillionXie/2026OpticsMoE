"""Pinned official LCM-LoRA tail candidate; Qwen remains the only input encoder.

Pretrained LoRA is fused into the SAME SDXL UNet before the task's paired LoRA is
installed. Its gain is a tail/sampler gain, never attributed solely to optics.
https://huggingface.co/latent-consistency/lcm-lora-sdxl
https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from .white_reference_scene_probe import digest,write

REPO='latent-consistency/lcm-lora-sdxl'
REVISION='a18548dd4956b174ec5b0d78d340c8dae0a129cd'
WEIGHT='pytorch_lora_weights.safetensors'
WEIGHT_SHA='a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141'


def checked_tail(root):
    report=json.loads((root/'asset_report.json').read_text())
    if not (report['complete'] and report['repo_id']==REPO and report['revision']==REVISION and report['license']=='openrail++'):
        raise ValueError('Unverified fast-tail asset identity')
    if report['weight_sha256']!=WEIGHT_SHA or digest(root/WEIGHT)!=WEIGHT_SHA:raise ValueError('Changed fast-tail weights')
    return dict(kind='official_LCM_LoRA_SDXL_v1',repo_id=REPO,revision=REVISION,weight_sha256=WEIGHT_SHA,
        scheduler='LCMScheduler',CFG=1.,pretrained_LoRA_fused_and_unloaded_before_task_LoRA=True,
        Qwen_only_RGB_text_entry_unchanged=True,not_an_optical_only_speed_gain=True)


def install_tail(pipe,root):
    from diffusers import LCMScheduler
    descriptor=checked_tail(root)
    before={n:tuple(p.shape) for n,p in pipe.unet.named_parameters()}
    pipe.load_lora_weights(str(root),weight_name=WEIGHT,adapter_name='scene_fast_lcm',local_files_only=True)
    pipe.fuse_lora(adapter_names=['scene_fast_lcm'],safe_fusing=True)
    # Removing the merged adapter layers before task-LoRA insertion is necessary:
    # a partly merged PEFT layer can otherwise bypass another unmerged adapter.
    pipe.unload_lora_weights()
    assert not any('lora_' in n for n,p in pipe.unet.named_parameters())
    assert before=={n:tuple(p.shape) for n,p in pipe.unet.named_parameters()},'Fast adapter changed the deployed UNet structure'
    pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
    return descriptor


def download(output):
    from huggingface_hub import snapshot_download
    import requests
    if (output/'asset_report.json').exists():return checked_tail(output)
    if output.exists():raise ValueError('Keep incomplete asset evidence; use an explicit new retry directory')
    # The official JSON uses stable field names across the workstation's older
    # and the server's modern huggingface_hub ModelInfo dataclass variants.
    response=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=60);response.raise_for_status();meta=response.json()
    assert meta['sha']==REVISION and meta['cardData']['license']=='openrail++'
    official={r['rfilename']:r for r in meta['siblings']}
    assert official[WEIGHT]['lfs']['sha256']==WEIGHT_SHA
    output.mkdir(parents=True)
    write(output/'download_execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__)),GPU_started=False))
    snapshot_download(REPO,revision=REVISION,allow_patterns=[WEIGHT,'README.md'],local_dir=output)
    assert digest(output/WEIGHT)==WEIGHT_SHA and (output/WEIGHT).stat().st_size==official[WEIGHT]['size']
    data=(output/'README.md').read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==official['README.md']['blobId']
    write(output/'asset_report.json',dict(complete=True,repo_id=REPO,revision=REVISION,license='openrail++',weight_sha256=WEIGHT_SHA,
        weight_bytes=official[WEIGHT]['size'],model_card_git_blob=blob,download_only=True,training_or_generation_performed=False))
    return checked_tail(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    print(json.dumps(download(a.output),indent=2))


if __name__=='__main__':main()
