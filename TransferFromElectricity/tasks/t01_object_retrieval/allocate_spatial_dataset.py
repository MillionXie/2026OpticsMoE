"""Wait for a free RTX without taking a card between another suite's runs.

The allocator may run from a newer commit than the immutable training checkout.
It records both source identities and launches the old checkout's suite command.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


def reserved_gpus(directory,prefix):
    result=set()
    for path in directory.glob(f'{prefix}_*_execution.json'):
        try:audit=json.loads(path.read_text())
        except json.JSONDecodeError:return None  # A writer is mid-update: wait conservatively.
        if not audit['complete']:result.update(row['gpu_uuid'] for row in audit['records'])
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--training-sha',required=True)
    parser.add_argument('--controller-sha',required=True)
    parser.add_argument('--candidate-uuids',nargs='+',required=True)
    parser.add_argument('--dataset',choices=['caltech','cifar','imagenette'],required=True)
    parser.add_argument('--methods',nargs='+',required=True)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--reservation-prefix',action='append',help='Also avoid GPUs reserved by these experiment suites')
    parser.add_argument('--config')
    parser.add_argument('--run-kind',choices=['simulation','smoke'],default='simulation')
    args=parser.parse_args();root=args.training_root.resolve()
    git=lambda *values:subprocess.check_output(['git',*values],cwd=root,text=True).strip()
    if git('rev-parse','HEAD')!=args.training_sha:raise ValueError('Training checkout differs from locked SHA')
    if git('status','--short') not in {'','?? data'}:raise ValueError('Unexpected changes in training checkout')
    if git('rev-parse',args.controller_sha)!=args.controller_sha:raise ValueError('Use the full controller commit SHA')
    devices=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).splitlines()))
    for uuid in args.candidate_uuids:
        names=[r[1].strip() for r in devices if r[0].strip()==uuid]
        if not uuid.startswith('GPU-') or len(names)!=1 or 'RTX' not in names[0] or 'A100' in names[0]:
            raise ValueError(f'Forbidden or unknown GPU: {uuid}')
    runs=root/'TransferFromElectricity/tasks/t01_object_retrieval/runs'
    directory=runs/args.run_kind;directory.mkdir(parents=True,exist_ok=True)
    if args.config and not (root/args.config).is_file():raise FileNotFoundError(args.config)
    stem=f'{args.prefix}_{args.dataset}'
    if list(directory.glob(stem+'_*_s42')) or (directory/(stem+'_execution.json')).exists():
        raise FileExistsError('This dataset has already started')
    allocation=directory/(stem+'_allocation.json')
    if allocation.exists():raise FileExistsError(allocation)
    previous=None
    while True:
        reservations=[reserved_gpus(runs/kind,prefix) for kind in ('simulation','smoke')
                      for prefix in [args.prefix,*(args.reservation_prefix or [])]]
        reserved=None if any(r is None for r in reservations) else set().union(*reservations)
        rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()))
        busy={u:sorted(int(r[0]) for r in rows if r[1].strip()==u) for u in args.candidate_uuids}
        state=(busy,None if reserved is None else sorted(reserved))
        if state!=previous:print('WAIT occupied/reserved',state,flush=True);previous=state
        eligible=[] if reserved is None else [u for u in args.candidate_uuids if not busy[u] and u not in reserved]
        if eligible:break
        time.sleep(5)
    uuid=eligible[0]
    command=[sys.executable,'-u','-m','TransferFromElectricity.tasks.t01_object_retrieval.run_spatial_suite',
             '--gpu-uuids',uuid,'--datasets',args.dataset,'--methods',*args.methods,'--prefix',args.prefix,'--wait-for-gpus','--run-kind',args.run_kind]
    if args.config:command+=['--config',args.config]
    record={'controller_sha':args.controller_sha,'training_sha':args.training_sha,'training_root':str(root),
            'candidate_uuids':args.candidate_uuids,'chosen_uuid':uuid,'command':command,'selected_at':time.time(),
            'policy':'Choose an empty RTX, excluding all GPUs reserved by incomplete paired suites','status':'running'}
    with (directory/(stem+'_allocated_queue.log')).open('x') as stream:
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record['queue_pid']=process.pid;allocation.write_text(json.dumps(record,indent=2))
        print('SELECTED',uuid,'queue PID',process.pid,flush=True)
        code=process.wait()
    record.update(status='complete' if code==0 else 'failed',returncode=code,finished_at=time.time())
    allocation.write_text(json.dumps(record,indent=2))
    if code:raise RuntimeError('Allocated dataset suite failed; logs preserved')


if __name__=='__main__':main()
