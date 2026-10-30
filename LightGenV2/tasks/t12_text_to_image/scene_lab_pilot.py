"""Frozen TRAIN native prefix preparation/replay. No SDK, training or FFT fallback."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from .scene_lab_protocol import Ledger, STAGES, NeedCapture, array_sha, put, sha
from .scene_lab_entry import train_requests
from .scene_lab_native import NativeRunner, load_head, F9, F6


def mark_input_ready(folder):
    import numpy as np
    folder=Path(folder);ticket=json.loads((folder/'ticket.json').read_text())
    keys=dict(amplitude='amplitude_sha256',total_phase='total_phase_sha256',amplitude_code='logical_amplitude_code_sha256',phase_code='logical_phase_code_sha256')
    with np.load(folder/'logical_fields.npz',allow_pickle=False)as z:
        if set(z.files)!=set(keys)or any(array_sha(z[k])!=ticket[v]for k,v in keys.items()):raise ValueError('Incomplete/changed input payload')
    marker=folder/'input.ready'
    if not marker.exists():marker.touch(exist_ok=False)
    return marker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','infer'),required=True)
    p.add_argument('--stage',choices=STAGES)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--session',type=Path)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--runtime-source-commit',help='Execution source identity; frozen ledger/model identity remains unchanged')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=2)
    p.add_argument('--gpu-memory-fraction',type=float)
    p.add_argument('--request-pause-ms',type=float,default=0)
    p.add_argument('--ready-markers',action='store_true')
    a=p.parse_args()
    import torch
    from .scene_paired_render import install_optical_frontend
    from .qwen3_scene_paired_train import new_qwen
    torch.set_num_threads(2)
    if a.gpu_memory_fraction is not None:
        if not 0<a.gpu_memory_fraction<=.5:raise ValueError('Explicit memory cap must be in (0,.5]')
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,device=0)
    if not 0<=a.request_pause_ms<=1000:raise ValueError('Invalid request pacing')
    task=Path(__file__).resolve().parent;base=task/'runs/simulation'
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    put(a.out/'execution.json',dict(runtime_source_commit=a.runtime_source_commit or a.source_commit,ledger_model_source_commit=a.source_commit,source_SHA256=sha(__file__),SDK=False))
    requests=train_requests(base/'scene_natural_TRAIN608_condition_cache_20261030/TRAIN_input_manifest.json',base/'scene_two_sentence_pairs_20261029',a.count)
    if (a.ledger/'contract.json').exists():
        contract=json.loads((a.ledger/'contract.json').read_text());session=contract['session']
        if contract['requests']!=requests:raise ValueError('Fixed TRAIN pilot changed')
    else:
        if a.session is None:raise ValueError('Reviewed session required')
        session=json.loads(a.session.read_text())
    if session.get('scope')!='reviewed_physical':raise ValueError('Synthetic session is not physical pilot')
    ledger=Ledger(a.ledger,session,dict(F9=F9,F6=F6,source_commit=a.source_commit),requests,create=not (a.ledger/'contract.json').exists())
    if a.mode=='prepare':
        if a.stage is None:raise ValueError('Pending stage required')
        admission=ledger.admit_stage(a.stage)
    else:
        for stage in STAGES:
            for request in requests:ledger.receipt(stage,request)
    qwen,processor=new_qwen(SimpleNamespace(qwen=a.qwen,qwen_side=256))
    info=install_optical_frontend(qwen,base/'qwen3_scene_LG_frontend_20261029',base/'qwen3_scene_LG_teacher_features_20261029',base/'scene_natural_TRAIN608_condition_cache_20261030/LG')
    if info['checkpoint_sha256']!=F6:raise ValueError('Wrong frontend')
    head,_=load_head(base/'scene_natural_single_EG_joint12_training_20261030/best_checkpoint.pt',expected_sha256=F9)
    runner=NativeRunner(qwen,processor,head.cuda(),base/'scene_two_sentence_pairs_20261029',frontend_sha256=F6,head_sha256=F9)
    if a.mode=='prepare':
        prepared=0
        def streamed(request,provider):
            nonlocal prepared
            try:return runner(request,provider)
            except NeedCapture:
                folder=ledger.folder(a.stage,request)
                if a.ready_markers:
                    # Empty marker is created only after the native runner has
                    # unwound and closed the complete payload/ticket files.
                    mark_input_ready(folder)
                prepared+=1
                if prepared%16==0 or prepared==len(requests):print(json.dumps(dict(stage=a.stage,prepared=prepared,total=len(requests))),flush=True)
                if a.request_pause_ms:
                    import time
                    time.sleep(a.request_pause_ms/1000)
                raise
        pending=ledger.prepare(a.stage,streamed,admission=admission)
        put(a.out/'report.json',dict(complete=True,mode=a.mode,stage=a.stage,pending=pending,TRAIN=len(requests),TRAIN_only=True,TEST_used=False,SDK=False,FFT_fallback=False,ledger_sha256=ledger.key,gpu_memory_fraction=a.gpu_memory_fraction,request_pause_ms=a.request_pause_ms))
    else:
        rows=[]
        for request in requests:
            result=ledger.infer(request,runner)
            file=a.out/(request['sample_id']+'_conditions.pt')
            torch.save({k:v.detach().cpu()for k,v in result['conditions'].items()},file)
            rows.append(dict(sample_id=request['sample_id'],conditions_file=file.name,conditions_sha256=sha(file)))
        put(a.out/'report.json',dict(complete=True,mode=a.mode,rows=rows,TRAIN_only=True,TEST_used=False,SDK=False,FFT_fallback=False,ledger_sha256=ledger.key,
            native_forward_attempts=runner.native_forward_attempts,completed_Qwen_forwards=runner.completed_forwards,SDXL_images_generated=0))


if __name__=='__main__':main()
