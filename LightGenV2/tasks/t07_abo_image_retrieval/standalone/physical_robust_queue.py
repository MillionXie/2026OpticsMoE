"""Bounded two-GPU continuation from the pinned deployed checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',nargs=2,required=True);a=p.parse_args()
    task=a.root/'LightGenV2/tasks/t07_abo_image_retrieval';runs=task/'runs/simulation'
    assets=a.root/'.codex_tmp/t07_robust_assets_20260926'
    if not assets.exists():
        assets.mkdir()
        with zipfile.ZipFile(task/'releases/abo_i2i_83125_20260917.zip') as z:
            for member in z.namelist():
                if member.startswith('assets/') and not member.endswith('/'):
                    target=assets/Path(member).relative_to('assets')
                    if not target.resolve().is_relative_to(assets.resolve()):raise ValueError('Unsafe archive path')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
    checkpoint=assets/'best.pt'
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha=='c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0'
    def command(profile,output,smoke=False):
        return [sys.executable,'-u','-m','LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt',
            '--data',str(a.root/'data/abo_similarity10_data'),
            '--manifest',str(runs/'abo200_enrolled_protocol_20260913/protocol.json'),
            '--assets',str(assets),'--checkpoint',str(checkpoint),'--expected-checkpoint-sha256',sha,
            '--multi-view','--refine-profile',profile,'--lr-scale','.2',
            '--epochs','1' if smoke else '20','--steps','2' if smoke else '100',
            '--eval-every','1' if smoke else '5','--batch-size','4','--bank-batch-size','16',
            '--output',str(output)]
    state_path=runs/'physical_robust35_queue_20260926.json'
    if state_path.exists():raise FileExistsError('Queue already launched; inspect existing processes, never duplicate')
    state=dict(status='smoke',pid=os.getpid(),checkpoint_sha256=sha,children=[])
    def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus[0],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    smoke=task/'runs/smoke/physical_robust35_20260926'
    smoke.parent.mkdir(parents=True,exist_ok=True)
    try:
        with smoke.with_suffix('.log').open('wb') as log:
            subprocess.run(command('physical_robust35',smoke,True),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        children=[]
        for profile,gpu in zip(('physical_robust35','physical_robust35_no_shift'),a.gpus):
            output=runs/(profile+'_20260926')
            cmd=command(profile,output)
            log=output.with_suffix('.log').open('wb')
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu))
            log.close();children.append(proc)
            state['children'].append(dict(profile=profile,pid=proc.pid,gpu=gpu,command=cmd))
        state['status']='training';save()
        exits=[proc.wait() for proc in children]
        state.update(status='complete' if exits==[0,0] else 'failed',exit_codes=exits);save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
