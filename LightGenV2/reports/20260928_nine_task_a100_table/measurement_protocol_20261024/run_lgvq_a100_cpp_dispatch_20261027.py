"""Guarded A100 C++ host-dispatch candidate; original raw adapter and clock gates."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import run_lgvq_a100_patch_parallel_20261027 as core
import lgvq_async_worker_20261026 as native
import lgvq_a100_cpp_dispatch_20261027 as dispatch

ENTRY=Path(__file__).resolve();CORE_ENTRY=core.ENTRY;SOURCES=core.source_map


def source_map():
    result=SOURCES()
    for path in (CORE_ENTRY,ENTRY,ENTRY.with_name('lgvq_a100_cpp_dispatch_20261027.py'),
                 ENTRY.with_name('lgvq_a100_cpp_dispatch_20261027.cpp'),ENTRY.with_name('build_lgvq_cpp_dispatch_20261027.py')):
        result[path.relative_to(core.original.REPO).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def cpp_worker(args):
    dispatch.load_extension()
    previous=native.worker;namespace=dict(previous.__globals__)
    exec(compile(dispatch.worker_AST(inspect.getsource(previous)),str(native.__file__)+'::cpp_dispatch','exec'),namespace)
    native.worker=namespace['worker']
    try:return core.parallel_worker(args)
    finally:native.worker=previous


@contextmanager
def configured():
    saved={k:getattr(core,k) for k in ('ENTRY','source_map')}
    core.ENTRY=ENTRY;core.source_map=source_map
    try:
        with core.configured():
            core.original.worker=cpp_worker
            yield
    finally:
        for k,v in saved.items():setattr(core,k,v)


if __name__=='__main__':
    with configured():core.original.main()
