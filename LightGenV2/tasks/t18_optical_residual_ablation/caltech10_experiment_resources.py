"""Human-authorized four non-A100 slots, shared across the new T18 campaign."""
from pathlib import Path
import caltech10_gap as parent
GPU={0:'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',**parent.GPU}
A100=parent.s.ALLOWED[6]
def slots():
    jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(parent.HERE) not in cmd:continue
            gpu=next((v.decode().split('=',1)[1] for v in (p/'environ').read_bytes().split(b'\0') if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu in set(GPU.values())|{A100}:jobs.append(dict(pid=int(p.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    counts={g:sum(j['gpu_uuid']==g for j in jobs) for g in set(GPU.values())|{A100}}
    assert counts[A100]==0 and len(jobs)<=4 and max(counts.values(),default=0)<=1,jobs
    return dict(jobs=jobs,counts=counts,total=len(jobs),maximum=4,per_card_maximum=1,allowed_physical_cards=[0,2,4,5],a100_prohibited=True)
