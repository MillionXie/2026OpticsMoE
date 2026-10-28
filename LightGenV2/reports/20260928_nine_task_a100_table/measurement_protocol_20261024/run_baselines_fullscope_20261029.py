"""Cold 512-video lab baseline remainder, after resize-inclusive LG completes.

Native model/preprocessing/arithmetic, B1, no inference warmup, full file-input
to CPU score wall, original power/identity/exclusion guards. Reuse the accepted
Qwen50 triple; only Qwen100 and GLM/CLIP/YOLO50/100 each three fresh rounds run.
No DS/Kimi/rental or existing-run mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ENTRY=Path(__file__).resolve();HERE=ENTRY.parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
import run_temporal_qwen_fullscope_20261029 as adapter

base=adapter.base;ORIGINAL_CONFIG=adapter.ORIGINAL_CONFIG;ORIGINAL_SOURCES=adapter.ORIGINAL_SOURCES
PROTOCOL='lab_baselines_cold_file_to_CPU_B1_512x3_20261029'
SIM=base.TASK/'runs/simulation';GPU=adapter.GPU

def source_map():
    paths=[ENTRY,Path(adapter.__file__)]
    return {**ORIGINAL_SOURCES(),**{p.relative_to(REPO).as_posix():base.sha(p) for p in paths}}

def missing_plan(profile,models=None,caps=None,repeats=None):
    if profile!='lab':raise ValueError('No rental or DS/Kimi authorization in this run')
    return [('qwen',100,r) for r in (1,2,3)]+[(m,c,r) for m in ('glm','clip','yolo') for c in (50,100) for r in (1,2,3)]

def config():
    original=ORIGINAL_CONFIG()
    return dict(original,protocol_id=PROTOCOL,videos_per_repeat=512,
        authorization='2026-10-29 user: resize-inclusive LG first, then remaining lab baselines MPS50/100',
        resource_caps=[50,100],models=['qwen','glm','clip','yolo'],
        retained='qwen_fullscope_20261029_02 Qwen50 triple; no rerun',
        inference_warmup=0,loading_is_outside_main_window=True)

def configure(root):
    adapter.ENTRY=ENTRY;adapter.NEW_PROTOCOL=PROTOCOL;adapter.source_map=source_map
    adapter.configure(root)
    base.plan=missing_plan;base.config=config
    for fn in (base.summarize,base.audit_point,base.worker,base.suite):
        fn.__globals__.update(vars(base));fn.__globals__.update(ENTRY=ENTRY,NEW_PROTOCOL=PROTOCOL)

def verify_predecessor(root):
    root=root.resolve()
    if root.parent!=SIM or not re.fullmatch(r'lgvq_compute_scope_20261029_\d{2}',root.name):
        raise ValueError('Explicit latest resize-inclusive LG root required')
    state=base.read(root/'suite_status.json')
    if state['status']!='complete' or not state['mps_released'] or state['completed']!=[f'lg_repeat{i:02d}' for i in (1,2,3)]:
        raise ValueError('LG triple must finish before any baseline begins')
    for key in ('pid','child_pid'):
        if Path('/proc',str(state[key])).exists():raise ValueError('Predecessor process still exists')
    for name in state['completed']:
        report=base.read(root/name/'compute_report.json')
        if not report.get('CPU_resize_and_normalization_included') or not report.get('all_predictions_bitwise_equal'):
            raise ValueError('Only actual resize-inclusive numerically qualified LG predecessor allowed')
    result=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY='/DATA/DATA1/guest3/.cache/'+root.name),timeout=5)
    if result.returncode==0:raise ValueError('LG private MPS not released')

def retained_qwen50():
    root=SIM/'qwen_fullscope_20261029_02';state=base.read(root/'suite_status.json')
    if state['status']!='complete' or not state['mps_released']:raise ValueError('Accepted Qwen50 triple required')
    points=[]
    for repeat in (1,2,3):
        leaf=root/f'qwen_cap50_repeat{repeat:02d}';audit=base.read(leaf/'cpu_audit.json');report=base.read(leaf/'report.json')
        if audit['status']!='passed' or audit['measured_videos']!=512 or report['cap']!=50 or report['batch']!=1:
            raise ValueError('Wrong retained Qwen50 scope')
        for name,key in [('report.json','report_sha256'),('calls.csv','calls_sha256'),('telemetry.csv','power_sha256')]:
            if base.sha(leaf/name)!=audit[key]:raise ValueError('Retained original evidence changed')
        points.append(dict(path=str(leaf),audit_sha256=base.sha(leaf/'cpu_audit.json'),report_sha256=base.sha(leaf/'report.json')))
    return points

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('suite','worker','audit'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suite-root',type=Path)
    p.add_argument('--after-lg',type=Path);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--deadline',type=float);p.add_argument('--profile',default='lab',choices=('lab',))
    p.add_argument('--model',choices=('qwen','glm','clip','yolo'));p.add_argument('--cap',type=int,choices=(50,100))
    p.add_argument('--repeat',type=int,choices=(1,2,3));p.add_argument('--model-path',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--reproduction-receipt',type=Path)
    p.add_argument('--gpu-uuid',default=GPU,choices=(GPU,));p.add_argument('--pipe')
    args=p.parse_args();args.output=args.output.resolve();root=(args.suite_root or args.output).resolve()
    if root.parent!=SIM or not re.fullmatch(r'baselines_fullscope_20261029_\d{2}',root.name):raise ValueError('New canonical lab baseline root required')
    if args.phase!='suite' and args.output.parent!=root:raise ValueError('Exact worker leaf required')
    args.suite_root=root;args.pipe=args.pipe or '/DATA/DATA1/guest3/.cache/'+root.name
    args.models=['qwen','glm','clip','yolo'];args.caps=[50,100];args.repeats=[1,2,3]
    if args.phase!='audit' and (args.deadline is None or not 0<args.deadline-time.time()<=28800):
        raise ValueError('Finite maximum eight-hour bound required')
    if args.phase=='worker' and (args.model,args.cap,args.repeat) not in missing_plan('lab'):
        raise ValueError('Only the fixed missing21 points may run')
    configure(root)
    if args.phase=='audit':print(json.dumps(base.audit_point(args.output,args.manifest),ensure_ascii=True));return
    commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT']
    for name,digest in source_map().items():
        blob=subprocess.check_output(['git','show',commit+':'+name.replace('\\','/')],cwd=REPO,timeout=10)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published baseline source differs: '+name)
    if args.phase=='suite':
        if args.after_lg is None:raise ValueError('Explicit completed LG predecessor required')
        verify_predecessor(args.after_lg);retained=retained_qwen50()
        # Keep retention receipt outside the not-yet-created native suite root.
        receipt=root.with_name(root.name+'.retained_qwen50.json')
        with receipt.open('x',encoding='utf8') as f:json.dump(retained,f,indent=2)
    (base.suite if args.phase=='suite' else base.worker)(args)

if __name__=='__main__':main()
