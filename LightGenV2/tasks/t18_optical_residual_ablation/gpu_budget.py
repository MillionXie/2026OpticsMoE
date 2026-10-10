"""T18 explicit spare-memory execution; no changes to other processes."""
import hashlib
from pathlib import Path
import subprocess

ALLOWED={'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
         'GPU-1b963983-7909-af6e-0528-f0f0661ab549'}


def preflight(gpu,shared=False,memory_gib=3.):
    assert gpu in ALLOWED and 1<=memory_gib<=4
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True).splitlines()
    info=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total',
        '--format=csv,noheader,nounits'],text=True)
    fields=next(line.split(',') for line in info.splitlines() if line.startswith(gpu))
    free,total=int(fields[1]),int(fields[2])
    if not shared:assert gpu not in [x.strip() for x in occupied], 'GPU occupied'
    assert free>=memory_gib*1024+800,'Insufficient spare memory for bounded execution'
    return dict(gpu_uuid=gpu,spare_memory_authorized=shared,memory_limit_gib=memory_gib,
        free_mib_at_start=free,total_mib=total,other_process_on_gpu=gpu in [x.strip() for x in occupied],
        safety_margin_mib=800,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def configure(torch,gpu,shared=False,memory_gib=3.):
    policy=preflight(gpu,shared,memory_gib)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(memory_gib*1024**3/total,0)
    return policy
