"""Own non-A100 scheduling guard; frozen parent modules remain unmodified."""
from pathlib import Path

ALL_GPU={0:'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',1:'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',2:'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',3:'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd',4:'GPU-1b963983-7909-af6e-0528-f0f0661ab549',5:'GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb',6:'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'}
GPU={i:ALL_GPU[i] for i in (0,1,2,5)}

def validate(jobs):
    assert all(j['gpu_uuid'] in ALL_GPU.values() and j['gpu_uuid']!=ALL_GPU[6] for j in jobs),jobs
    assert len({j['pid'] for j in jobs})==len(jobs),jobs
    counts={g:sum(j['gpu_uuid']==g for j in jobs) for g in ALL_GPU.values()}
    assert len(jobs)<=4 and max(counts.values(),default=0)<=1,jobs
    return dict(total=len(jobs),counts=counts,jobs=jobs)

def slots():
    here=str(Path(__file__).resolve().parent);jobs=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            cmd=path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if here not in cmd:continue
            gpu=next((x.decode().split('=',1)[1] for x in path.joinpath('environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu:jobs.append(dict(pid=int(path.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    return validate(jobs)

def proof():
    def jobs(indices):return [dict(pid=100+i,gpu_uuid=ALL_GPU[g]) for i,g in enumerate(indices)]
    assert validate([])['total']==0 and validate(jobs([0,1,2,5]))['total']==4
    assert validate(jobs([3,4]))['total']==2
    invalid=[jobs([6]),jobs([0,0]),jobs([0,1,2,3,5]),[dict(pid=100,gpu_uuid='unknown')],jobs([2,2]),jobs([5,5]),[dict(pid=100,gpu_uuid=ALL_GPU[0]),dict(pid=100,gpu_uuid=ALL_GPU[1])]]
    for case in invalid:
        try:validate(case)
        except AssertionError:continue
        raise AssertionError(('resource guard accepted invalid jobs',case))
    return dict(passed=True,accepted_cases=3,rejected_cases=len(invalid),allowed_launch_order=list(GPU),A100_allowed=False,max_children=4,one_child_per_gpu=True,foreign_process_actions=0)
