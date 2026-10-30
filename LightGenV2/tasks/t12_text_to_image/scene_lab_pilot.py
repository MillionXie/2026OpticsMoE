"""Frozen TRAIN2 native prefix preparation/replay. No SDK, training or FFT fallback."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from .scene_lab_protocol import Ledger, STAGES, put, sha
from .scene_lab_entry import train_requests
from .scene_lab_native import NativeRunner, load_head, F9, F6


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','infer'),required=True)
    p.add_argument('--stage',choices=STAGES)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--session',type=Path)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    import torch
    from .scene_paired_render import install_optical_frontend
    from .qwen3_scene_paired_train import new_qwen
    torch.set_num_threads(2)
    task=Path(__file__).resolve().parent;base=task/'runs/simulation'
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    requests=train_requests(base/'scene_natural_TRAIN608_condition_cache_20261030/TRAIN_input_manifest.json',base/'scene_two_sentence_pairs_20261029',2)
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
        ledger.ready(a.stage)
    else:
        for stage in STAGES:
            for request in requests:ledger.receipt(stage,request)
    qwen,processor=new_qwen(SimpleNamespace(qwen=a.qwen,qwen_side=256))
    info=install_optical_frontend(qwen,base/'qwen3_scene_LG_frontend_20261029',base/'qwen3_scene_LG_teacher_features_20261029',base/'scene_natural_TRAIN608_condition_cache_20261030/LG')
    if info['checkpoint_sha256']!=F6:raise ValueError('Wrong frontend')
    head,_=load_head(base/'scene_natural_single_EG_joint12_training_20261030/best_checkpoint.pt',expected_sha256=F9)
    runner=NativeRunner(qwen,processor,head.cuda(),base/'scene_two_sentence_pairs_20261029',frontend_sha256=F6,head_sha256=F9)
    if a.mode=='prepare':
        pending=ledger.prepare(a.stage,runner)
        put(a.out/'report.json',dict(complete=True,mode=a.mode,stage=a.stage,pending=pending,TRAIN_only=True,TEST_used=False,SDK=False,FFT_fallback=False,ledger_sha256=ledger.key))
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
