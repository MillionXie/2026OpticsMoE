"""Isolated original-A100 Conv3d storage-layout candidate.

Reuse the original guarded 2048-call entry and all math/timing gates. Only the
native learned Conv3d weight/input storage layout changes; no GEMM, folded
geometry, dtype/weight change, warmup or scope correction. The wrapper and
layout adapter join the exact published frozen source closure.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import subprocess

import run_lgvq_async_20261026 as original
from lgvq_development_conv_layout_20261027 import install

ENTRY=Path(__file__).resolve()
ORIGINAL_ENTRY=original.ENTRY
ORIGINAL_SOURCES=original.source_map
ORIGINAL_WORKER=original.worker
ORIGINAL_UNIQUE=original.assert_unique_controller


def source_map():
    result=ORIGINAL_SOURCES()
    for p in (ORIGINAL_ENTRY,ENTRY,ENTRY.with_name('lgvq_development_conv_layout_20261027.py')):
        result[p.relative_to(original.REPO).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def verify_exact_published(commit,sources):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact published candidate required')
    for name,digest in sources.items():
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=original.REPO,timeout=10)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Frozen original/layout source differs: '+name)


def assert_unique():
    # Check the unchanged original entry as well as the current wrapper entry.
    current=original.ENTRY
    try:
        original.ENTRY=ORIGINAL_ENTRY
        ORIGINAL_UNIQUE()
    finally:original.ENTRY=current
    ORIGINAL_UNIQUE()


def layout_worker(args):
    import lgvq_live_frontend as live
    previous=live.LiveFront.__init__
    def init(front,*a,**kw):
        previous(front,*a,**kw)
        info=install(front)
        info.update(hardware_scope='ORIGINAL_A100_250W_108SM',original_A100_reference_retained=True,
            GEMM_or_geometry_replacement=False,all_original_math_and_timing_gates_retained=True)
        original.write(args.output/'a100_layout_identity.json',info)
    live.LiveFront.__init__=init
    try:return ORIGINAL_WORKER(args)
    finally:live.LiveFront.__init__=previous


@contextmanager
def configured():
    saved={k:getattr(original,k) for k in ('ENTRY','source_map','verify_published_candidates','assert_unique_controller','worker')}
    original.ENTRY=ENTRY
    original.source_map=source_map
    original.verify_published_candidates=verify_exact_published
    original.assert_unique_controller=assert_unique
    original.worker=layout_worker
    try:yield
    finally:
        for k,v in saved.items():setattr(original,k,v)


if __name__=='__main__':
    with configured():original.main()
