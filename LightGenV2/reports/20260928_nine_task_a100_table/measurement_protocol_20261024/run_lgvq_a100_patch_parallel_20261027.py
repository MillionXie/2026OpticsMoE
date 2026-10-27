"""Isolated A100 native-Conv3d batch-parallel candidate; original guards intact."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import subprocess

import run_lgvq_async_20261026 as original
from lgvq_a100_patch_parallel_20261027 import install

ENTRY=Path(__file__).resolve()
ORIGINAL_ENTRY=original.ENTRY
ORIGINAL_SOURCES=original.source_map
ORIGINAL_WORKER=original.worker
ORIGINAL_UNIQUE=original.assert_unique_controller


def source_map():
    result=ORIGINAL_SOURCES()
    for p in (ORIGINAL_ENTRY,ENTRY,ENTRY.with_name('lgvq_a100_patch_parallel_20261027.py')):
        result[p.relative_to(original.REPO).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def verify_exact_published(commit,sources):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact published candidate required')
    for name,digest in sources.items():
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=original.REPO,timeout=10)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Frozen source differs: '+name)


def assert_unique():
    current=original.ENTRY
    try:
        for entry in (ORIGINAL_ENTRY,ENTRY,ENTRY.with_name('run_lgvq_a100_layout_20261027.py')):
            original.ENTRY=entry
            ORIGINAL_UNIQUE()
    finally:original.ENTRY=current


def parallel_worker(args):
    import lgvq_live_frontend as live
    previous=live.LiveFront.__init__;installed=[]
    def init(front,*a,**kw):
        previous(front,*a,**kw)
        info=install(front)
        info.update(hardware_scope='ORIGINAL_A100_250W_108SM',
            original_A100_reference_retained=True,all_original_math_and_timing_gates_retained=True)
        installed.append(info)
        original.write(args.output/'a100_patch_parallel_identity.json',info)
    live.LiveFront.__init__=init
    try:return ORIGINAL_WORKER(args)
    finally:
        live.LiveFront.__init__=previous
        # Post-window actual Python capture/eager invocation counts, including failures.
        if installed:original.write(args.output/'a100_patch_parallel_identity.json',installed[0])


@contextmanager
def configured():
    saved={k:getattr(original,k) for k in ('ENTRY','source_map','verify_published_candidates','assert_unique_controller','worker')}
    original.ENTRY=ENTRY;original.source_map=source_map
    original.verify_published_candidates=verify_exact_published
    original.assert_unique_controller=assert_unique;original.worker=parallel_worker
    try:yield
    finally:
        for k,v in saved.items():setattr(original,k,v)


if __name__=='__main__':
    with configured():original.main()
