"""Explicitly authorized three-point continuation, original deadline unchanged."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from . import measure_unified_inference_recheck as original

DEADLINE=1792312442.3183658
EXPECTED=['clip_native_b01_r01','clip_native_b01_r02','clip_native_b01_r03',
          'clip_native_b10_r01','clip_native_b10_r02','clip_native_b10_r03','qwen_paced_b03_r01']
REMAINING=[('glm',2,'paced',1),('clip',10,'paced',1),('yolo',3,'paced',1)]


def protected_files(base):
    folders=EXPECTED+['glm_paced_b02_r01']
    files=[p for name in folders for p in (base/name).rglob('*') if p.is_file()]
    files += [base/'suite_status.json']
    return {p.relative_to(base).as_posix():original.frozen.sha256(p) for p in files}


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','base','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    for key in ('manifest','base','output'):setattr(args,key,getattr(args,key).resolve())
    if args.deadline!=DEADLINE or time.time()>=DEADLINE:raise ValueError('Original deadline only')
    if args.output!=args.base/'resume03_20261018':raise ValueError('Canonical distinct continuation required')
    original.reproduction.require_new_output(args.output)
    previous=json.loads((args.base/'suite_status.json').read_text())
    if previous['status']!='failed' or previous['completed']!=EXPECTED:raise ValueError('Exactly7 complete original points required')
    sources=original.source_identity()
    if sources!=previous['source_sha256']:raise ValueError('Do not change original measurement implementation')
    protected=protected_files(args.base)
    for name in EXPECTED:
        report=json.loads((args.base/name/'report.json').read_text())
        if report['status']!='complete' or report['test_videos']!=558 or report['cost_test_videos']!=200:raise ValueError('Original point incomplete')
    source_sha=original.frozen.sha256(Path(__file__))
    c=original.config()
    with (original.TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.costs.wait_for_idle(args,c)
        if protected_files(args.base)!=protected:raise ValueError('Original evidence changed during admission')
        args.output.mkdir()
        state=dict(status='running',pid=os.getpid(),deadline=DEADLINE,completed=[],
            source_sha256=sources,continuation_source_sha256=source_sha,
            authorization='Human explicitly approved only3 remaining points within original16:34 window,2026-10-18',
            protected_original_sha256=protected,original_failed_suite_preserved=True)
        def save():original.frozen.write_json(args.output/'suite_status.json',state)
        save()
        try:
            for model,batch,kind,repeat in REMAINING:
                if time.time()>=DEADLINE:raise TimeoutError('Original finite deadline')
                original.reproduction.cuda_guard()
                name=f'{model}_{kind}_b{batch:02d}_r{repeat:02d}'
                state.update(phase=name,phase_started_unix=time.time());save()
                cmd=original.child_command(args,model,batch,kind,repeat,args.output/name)
                original.frozen.write_json(args.output/(name+'.command.json'),cmd)
                with (args.output/(name+'.log')).open('x') as stream:
                    subprocess.run(cmd,cwd=original.REPO,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=min(1200,DEADLINE-time.time()))
                report=json.loads((args.output/name/'report.json').read_text())
                if report['status']!='complete' or report['cost_test_videos']!=200:raise RuntimeError('Incomplete resumed point')
                state['completed'].append(name);save()
            original.reproduction.cuda_guard()
            if (protected_files(args.base)!=protected or original.source_identity()!=sources
                or original.frozen.sha256(Path(__file__))!=source_sha):raise RuntimeError('Protected source/evidence changed')
            state.update(status='complete',finished_unix=time.time())
        except BaseException as error:
            state.update(status='failed',error=repr(error));raise
        finally:save()


if __name__=='__main__':main()
