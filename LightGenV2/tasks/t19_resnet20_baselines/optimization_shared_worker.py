"""Resource-only entrypoint: permit two small T19 stages on a stable idle card.

The delegated scientific worker and its probe/model/loss hashes stay unchanged.
Actual wrapper command, source SHA and admission policy are added to provenance.
"""
import argparse
import os
from pathlib import Path
import sys
import time
from . import optimization_stage as stage
from .shared_idle_gpu import snapshot
from .train_easy100 import TASK,sharing_pids
from .frozen_resnet18 import file_sha256


def validate_shared(state, owned_pids, previous=None):
    registered=set(owned_pids)|set(state.get('t19_pids',[]))
    active=set(state['active']);foreign=active-registered
    free=state['total_mib']-state['used_mib']
    if len(registered)>=2:raise RuntimeError('Two T19 reservations already occupy this card')
    if free<6144 or state['utilization_percent']>50:
        raise RuntimeError('Need >=6GiB free and <=50 percent utilization')
    if foreign:
        if previous is None or state['observed_monotonic']-previous['observed_monotonic']<10:
            raise RuntimeError('Observe stable existing allocations for ten seconds')
        old_registered=set(owned_pids)|set(previous.get('t19_pids',[]))
        if (previous['utilization_percent']>50 or previous['total_mib']-previous['used_mib']<6144
                or set(previous['active'])-old_registered!=foreign
                or any(abs(previous['active'][p]-state['active'][p])>128 for p in foreign)):
            raise RuntimeError('Existing contexts are not stably idle')
    elif not active and state['used_mib']>100:
        raise RuntimeError('Unattributed allocation')
    return dict(policy='user_requested_two_T19_with_stable_idle_existing_contexts',
        foreign_pids=sorted(foreign),peer_pids=sorted(registered),free_mib=free,
        utilization_percent=state['utilization_percent'],max_T19_per_GPU=2,
        other_processes_modified=False,stable_observation_seconds=(state['observed_monotonic']-previous['observed_monotonic']) if foreign else 0)


def wait_shared(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise RuntimeError('Wrong CUDA UUID visibility')
    previous=None
    while True:
        state=snapshot(args.gpu_uuid)
        try:return validate_shared(state,sharing_pids(args),previous)
        except (RuntimeError,FileNotFoundError) as error:print('Waiting for low-utilization shared slot:',error,flush=True)
        if previous is None or state['observed_monotonic']-previous['observed_monotonic']>=15:previous=state
        time.sleep(5)


def main(args):
    original_write=stage.write_json
    def write(path,value):
        if Path(path).name=='run_manifest.json':
            value=dict(value,scientific_entrypoint_command=value.get('command'),
                command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],
                resource_entrypoint_SHA=file_sha256(Path(__file__)),
                resource_policy='max2_T19_free6GiB_util50_stable10s_original_four_cards')
        return original_write(path,value)
    stage.wait=wait_shared;stage.write_json=write
    stage.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('lsp','openmoji','salicon'),required=True)
    parser.add_argument('--phase',choices=('probe','formal'),required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--queue-status-path',type=Path,required=True)
    parser.add_argument('--probe-result',type=Path);parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--max-jobs-per-gpu',type=int,default=2)
    main(parser.parse_args())
