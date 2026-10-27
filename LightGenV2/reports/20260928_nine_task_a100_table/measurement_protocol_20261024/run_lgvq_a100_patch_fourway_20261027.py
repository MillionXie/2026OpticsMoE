"""One explicit four-way native-patch scheduling candidate, no automatic sweep."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import run_lgvq_a100_patch_parallel_20261027 as core

ENTRY=Path(__file__).resolve()
TWO_WAY_ENTRY=core.ENTRY
ORIGINAL_SOURCES=core.source_map
ORIGINAL_INSTALL=core.install


def source_map():
    result=ORIGINAL_SOURCES()
    result[TWO_WAY_ENTRY.relative_to(core.original.REPO).as_posix()]=hashlib.sha256(TWO_WAY_ENTRY.read_bytes()).hexdigest()
    return result


def fourway_install(front):return ORIGINAL_INSTALL(front,partitions=4)


@contextmanager
def configured():
    saved={k:getattr(core,k) for k in ('ENTRY','source_map','install')}
    core.ENTRY=ENTRY;core.source_map=source_map;core.install=fourway_install
    try:
        with core.configured():yield
    finally:
        for k,v in saved.items():setattr(core,k,v)


if __name__=='__main__':
    with configured():core.original.main()
