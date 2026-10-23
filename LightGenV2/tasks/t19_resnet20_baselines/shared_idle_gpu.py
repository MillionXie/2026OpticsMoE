"""Conservative sharing under the user's low-utilization parallelism approval.

No other process is signalled, reprioritized or modified. A shared card admits
at most one T19 stage with >=16GiB free and <=50% utilization, after its other
CUDA allocations have been observed stable for ten seconds.
"""
import os
from pathlib import Path
import subprocess
import time

from .train_easy100 import TASK, sharing_pids, validate_gpu_processes


def snapshot(uuid):
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    active = {}
    t19_pids = []
    for line in apps.splitlines():
        columns = [x.strip() for x in line.split(',')]
        if len(columns) == 3 and columns[0] == uuid:
            pid = int(columns[1])
            active[pid] = int(columns[2])
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                if ('LightGenV2.tasks.t19_resnet20_baselines.' in command
                        and str(TASK / 'runs') in command and uuid in command):
                    t19_pids.append(pid)
            except (FileNotFoundError, PermissionError):
                pass
    state = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    used, total, utilization = [int(x.strip()) for x in state.strip().split(',')]
    return dict(gpu_uuid=uuid, active=active, t19_pids=t19_pids, used_mib=used, total_mib=total,
                utilization_percent=utilization, observed_monotonic=time.monotonic())


def validate(state, owned_pids, previous=None):
    active = set(state['active'])
    registered = set(owned_pids) | set(state.get('t19_pids', []))
    owned = active & registered
    foreign = active - registered
    free = state['total_mib'] - state['used_mib']
    if foreign:
        if registered:
            raise RuntimeError('Shared card already has its one T19 stage')
        if free < 16384 or state['utilization_percent'] > 50:
            raise RuntimeError('Existing contexts do not have shared-idle headroom')
        if previous is None or state['observed_monotonic'] - previous['observed_monotonic'] < 10:
            raise RuntimeError('Observe existing allocation stability for ten seconds')
        if (previous['utilization_percent'] > 50
                or previous['total_mib'] - previous['used_mib'] < 16384):
            raise RuntimeError('Previous observation did not have shared-idle headroom')
        previous_t19 = set(owned_pids) | set(previous.get('t19_pids', []))
        if set(previous['active']) - previous_t19 != foreign:
            raise RuntimeError('Existing CUDA process set changed')
        if any(abs(state['active'][pid] - previous['active'][pid]) > 128 for pid in foreign):
            raise RuntimeError('Existing CUDA allocations are not stable')
    else:
        if len(owned) >= 2 or free < 6144 or state['utilization_percent'] > 95:
            raise RuntimeError('No room for another T19 stage')
        if not owned and state['used_mib'] > 100:
            raise RuntimeError('Unattributed allocation on empty card')
    # Existing T19 peers must still be genuine task workers on this exact card.
    own_apps = '\n'.join(f"{state['gpu_uuid']}, {pid}" for pid in owned)
    validate_gpu_processes(state['gpu_uuid'], own_apps, owned, max_jobs_per_gpu=2)
    return dict(policy='user_approved_low_utilization_parallelism', foreign_pids=sorted(foreign),
                owned_cuda_pids=sorted(owned), free_mib=free,
                utilization_percent=state['utilization_percent'],
                stable_observation_seconds=state['observed_monotonic'] - previous['observed_monotonic']
                    if foreign else 0, other_processes_modified=False)


def wait(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise RuntimeError('Worker must see only its requested UUID')
    previous = None
    while True:
        state = snapshot(args.gpu_uuid)
        try:
            return validate(state, sharing_pids(args), previous)
        except (RuntimeError, FileNotFoundError) as error:
            print('Waiting without CUDA under idle-sharing policy:', str(error), flush=True)
        # Compare a recent full ten-second observation, not a launch-time PID guess.
        if previous is None or state['observed_monotonic'] - previous['observed_monotonic'] >= 15:
            previous = state
        time.sleep(5)
