"""One owner, finite serial continuation of the approved remaining21 lab points.

Preserve the already-complete shared Qwen100 triple. Each child retains the
published T06 admission, private MPS, source/asset guards and independent audit.
No automatic retries, new rental, new model, changed precision or warming.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
MODULE="LightGenV2.tasks.t06_video_quality_assessment.measure_mps_e2e_20261024"
PIPE="/DATA/DATA1/guest3/.cache/t06_mps_e2e_20261024"


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def segments(resume):
    """The interruption had five qualified points and one zero-sample load."""
    if resume:
        return [("glm100_r3",["glm"],[100],[3]),
                ("glm50",["glm"],[50],[1,2,3]),
                ("clip_yolo",["clip","yolo"],[100,50],[1,2,3])]
    return [("qwen50",["qwen"],[50],[1,2,3]),
            ("glm_clip_yolo",["glm","clip","yolo"],[100,50],[1,2,3])]


def classify_processes(processes,servers,clients,worker,gpu):
    foreign=[row for row in processes if int(row[0].strip()) not in {*servers,worker} or row[1].strip()!=gpu]
    return dict(foreign_gpu_processes=foreign,foreign_mps_clients=sorted(set(clients)-{worker}))


def process_snapshot(gpu,worker,pipe=PIPE):
    raw=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-compute-apps=pid,gpu_uuid,used_memory',
                                 '--format=csv,noheader,nounits'],text=True,timeout=5)
    processes=[line.split(',') for line in raw.splitlines() if line.strip()]
    servers=set();clients=set()
    if processes:
        def control(command):
            result=subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,
                timeout=5,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=pipe))
            if result.returncode:raise RuntimeError('Private loading MPS query failed: '+result.stderr)
            return {int(value) for value in re.findall(r'^\s*(\d+)\s*$',result.stdout,re.M)}
        servers=control('get_server_list')
        for server in servers:clients|=control('get_client_list '+str(server))
    return dict(unix=time.time(),worker_pid=worker,processes=processes,
                servers=sorted(servers),clients=sorted(clients),**classify_processes(processes,servers,clients,worker,gpu))


def loading_snapshot(entry,suite):
    """Extra guard only during loading; timed-window worker guard is unchanged."""
    status=suite/'suite_status.json'
    if not status.exists():return None
    try:
        state=entry.read(status)
        if state.get('status')!='running':return None
        point=suite/state['phase'];worker_state=entry.read(point/'status.json')
    except (FileNotFoundError,json.JSONDecodeError):return None  # Keep the legacy status writer unchanged.
    if worker_state.get('status')!='loading':return None
    worker=int(state['child_pid'])
    if worker_state['pid']!=worker:raise ValueError('Loading worker identity differs')
    return dict(worker_state=worker_state,**process_snapshot(entry.config()['profiles']['lab']['gpu_uuid'],worker))


def terminate_own_loading_worker(snapshot,point,expected_module=MODULE):
    """Never signal a foreign process, MPS server or unverified process group."""
    worker=snapshot['worker_pid'];proc=Path('/proc')/str(worker)
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if (proc.stat().st_uid!=os.geteuid() or expected_module not in command or str(point) not in command
            or os.getpgid(worker)!=worker):raise RuntimeError('Refuse unverified worker signal')
    os.killpg(worker,signal.SIGTERM)


def wait_suite(child,entry,suite,root,name,deadline):
    observations=[]
    try:
        while True:
            if time.time()>=deadline:raise TimeoutError('Finite original suite deadline')
            try:return child.wait(timeout=min(1,max(.01,deadline-time.time())))
            except subprocess.TimeoutExpired:pass
            snapshot=loading_snapshot(entry,suite)
            if snapshot is None:continue
            observations.append(snapshot)
            if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                write(root/(name+'.foreign_loading_interruption.json'),snapshot)
                phase=entry.read(suite/'suite_status.json')['phase']
                terminate_own_loading_worker(snapshot,suite/phase)
                child.wait(timeout=40)  # Original suite owns private-MPS cleanup.
                raise RuntimeError('Foreign compute while loading; own worker stopped, preserve this point')
    finally:write(root/(name+'.loading_process_audit.json'),observations)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--deadline",type=float,required=True)
    p.add_argument("--resume-after-interruption",action="store_true",help="Human reauthorized only remaining16 points")
    args=p.parse_args()
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    root=args.output.resolve()
    if root.exists() or not root.is_relative_to(entry.CANONICAL):raise ValueError("New canonical continuation required")
    if not 0<args.deadline-time.time()<=14400:raise ValueError("Finite at-most-four-hour continuation")
    manifest=REPO/"experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv"
    references=[entry.CANONICAL/"qwen100_repaired01/qwen_cap100_repeat01",
                entry.CANONICAL/"qwen100_remaining01/qwen_cap100_repeat02",
                entry.CANONICAL/"qwen100_remaining01/qwen_cap100_repeat03"]
    # Independent CPU verification, not trusting names or mtime as provenance.
    verified=[entry.audit_point(path,manifest) for path in references]
    if any(row["status"]!="passed" for row in verified):raise ValueError("Shared Qwen100 incomplete")
    preserved=[]
    if args.resume_after_interruption:
        old=entry.CANONICAL/'lab_remaining01'
        if entry.read(old/'queue_status.json')['status']!='stopped':raise ValueError('Original queue is not stopped')
        paths=[old/'qwen50'/f'qwen_cap50_repeat{i:02d}' for i in (1,2,3)]
        paths += [old/'glm_clip_yolo'/f'glm_cap100_repeat{i:02d}' for i in (1,2)]
        preserved=[dict(path=str(path.relative_to(REPO)),**entry.audit_point(path,manifest)) for path in paths]
        interrupted=old/'glm_clip_yolo/glm_cap100_repeat03'
        if entry.read(interrupted/'status.json')['samples']!=0 or (interrupted/'report.json').exists():
            raise ValueError('Interrupted point is no longer the zero-sample loading run')
    frozen=entry.source_map()
    for path in [*references,*(paths if args.resume_after_interruption else [])]:
        if entry.read(path/'report.json')['sources']!=frozen:
            raise ValueError('Continuation runtime source differs from a qualified reference')
    root.mkdir(parents=True)
    state=dict(status="ready",pid=os.getpid(),deadline=args.deadline,completed=[],
               command=sys.argv,shared_qwen100=[dict(path=str(path.relative_to(REPO)),**v) for path,v in zip(references,verified)],
               source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
               protocol_id=entry.PROTOCOL,source_map=frozen)
    state['preserved_completed_points']=preserved
    state['planned_new_points']=16 if args.resume_after_interruption else 21
    state["controller_sha256"] = entry.sha(__file__)
    def save():write(root/"queue_status.json",state)
    save()
    try:
        for name,models,caps,repeats in segments(args.resume_after_interruption):
            if time.time()>=args.deadline-120:raise TimeoutError("No time for next original suite")
            if entry.source_map()!=frozen:raise ValueError("Frozen T06 source changed")
            command=[sys.executable,"-u","-m",MODULE,"--phase","suite","--models",*models,
                     "--caps",*[str(c) for c in caps],"--repeats",*[str(r) for r in repeats],"--manifest",str(manifest),
                     "--output",str(root/name),"--deadline",str(args.deadline)]
            write(root/(name+".command.json"),command)
            with (root/(name+".log")).open("x") as log:
                child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=entry.config()["profiles"]["lab"]["gpu_uuid"]))
                state.update(status="running",phase=name,child_pid=child.pid);save()
                try:code=wait_suite(child,entry,root/name,root,name,args.deadline)
                except BaseException:
                    if child.poll() is None:
                        # Worker has its own session. Stop only that verified
                        # worker first and let its original suite release MPS.
                        status=root/name/'suite_status.json'
                        if status.exists():
                            current=entry.read(status);pid=current.get('child_pid')
                            if pid and Path('/proc',str(pid)).exists():
                                terminate_own_loading_worker({'worker_pid':int(pid)},root/name/current['phase'])
                        try:child.wait(timeout=40)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20)
                    raise
            if code:raise RuntimeError(name+" failed; preserve raw evidence and stop")
            result=entry.read(root/name/"suite_status.json")
            if result["status"]!="complete" or not result["mps_released"]:raise ValueError("Child did not complete/release")
            state["completed"].append(name);save()
        state.update(status="complete",finished_unix=time.time())
    except BaseException as error:
        state.update(status="stopped",error=repr(error));raise
    finally:save()


if __name__=="__main__":main()
