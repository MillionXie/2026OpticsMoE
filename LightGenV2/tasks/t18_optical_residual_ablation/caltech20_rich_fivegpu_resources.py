"""2026-10-18 explicit five non-A100 GPU authorization; old guard frozen."""
from pathlib import Path
from caltech20_rich_pair50_depthwise_resources import ALL_GPU
GPU={i:ALL_GPU[i] for i in (0,1,2,3,4)}
def validate(jobs):
    assert len(jobs)<=5 and len({q['pid'] for q in jobs})==len(jobs),jobs
    assert all(q['gpu_uuid'] in GPU.values() for q in jobs),jobs
    counts={g:sum(q['gpu_uuid']==g for q in jobs) for g in ALL_GPU.values()}
    assert max(counts.values(),default=0)<=1,jobs
    return dict(total=len(jobs),counts=counts,jobs=jobs)
def slots():
    here=str(Path(__file__).resolve().parent);jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if here not in cmd:continue
            gpu=next((x.decode().split('=',1)[1] for x in p.joinpath('environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu:jobs.append(dict(pid=int(p.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    return validate(jobs)
def proof():
    jobs=lambda indices:[dict(pid=100+i,gpu_uuid=ALL_GPU[g]) for i,g in enumerate(indices)]
    assert validate(jobs([0,1,2,3,4]))['total']==5 and validate([])['total']==0
    bad=[jobs([0,0]),jobs([6]),jobs([5]),jobs([0,1,2,3,4,5]),[dict(pid=1,gpu_uuid='unknown')],[dict(pid=1,gpu_uuid=GPU[0]),dict(pid=1,gpu_uuid=GPU[1])]]
    for case in bad:
        try:validate(case)
        except AssertionError:continue
        raise AssertionError(case)
    return dict(passed=True,max_project_children=5,one_per_GPU=True,allowed_indices=list(GPU),A100=False,rejected=len(bad),foreign_process_actions=0)
