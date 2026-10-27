"""One isolated body-launch composition candidate on the two-stream patch front."""
import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import run_lgvq_a100_patch_parallel_20261027 as core
import lgvq_async_worker_20261026 as native
from lgvq_a100_batch_replay_20261027 import worker_AST

ENTRY=Path(__file__).resolve();CORE_ENTRY=core.ENTRY;ORIGINAL_SOURCES=core.source_map


def source_map():
    result=ORIGINAL_SOURCES()
    for path in (CORE_ENTRY,ENTRY,ENTRY.with_name('lgvq_a100_batch_replay_20261027.py'),
                 ENTRY.with_name('lgvq_a100_batch_raw_adapter_20261027.py')):
        result[path.relative_to(core.original.REPO).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def batch_worker(args):
    previous=native.worker
    tree,receipt=worker_AST(inspect.getsource(previous))
    namespace=dict(previous.__globals__)
    exec(compile(tree,str(native.__file__)+'::batch_imports','exec'),namespace)
    native.worker=namespace['worker']
    try:return core.parallel_worker(args)
    finally:
        native.worker=previous
        if args.output.exists():core.original.write(args.output/'a100_body_batch_identity.json',receipt)


@contextmanager
def configured():
    saved={k:getattr(core,k) for k in ('ENTRY','source_map')}
    core.ENTRY=ENTRY;core.source_map=source_map
    try:
        with core.configured():
            core.original.worker=batch_worker
            yield
    finally:
        for k,v in saved.items():setattr(core,k,v)


if __name__=='__main__':
    with configured():core.original.main()
