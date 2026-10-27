"""User-approved no-extra-warmup scope: electronic graph construction separate.

The actual worker/window and all original gates are unchanged. The separately
audited v3 ledger removes ONLY 44 witnessed capture_electronics host nodes.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import run_lgvq_a100_patch_parallel_20261027 as core

ENTRY=Path(__file__).resolve();CORE_ENTRY=core.ENTRY;SOURCES=core.source_map
PROFILE='A100_NO_EXTRA_WARMUP_ELECTRONIC_GRAPH_BUILD_SEPARATE_V3'


def source_map():
    result=SOURCES()
    for path in (CORE_ENTRY,ENTRY):
        result[path.relative_to(core.original.REPO).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def deployed_worker(args):
    core.original.write(args.output/'a100_deployed_scope_identity.json',dict(
        profile=PROFILE,explicit_extra_warmup_forwards=0,
        exclusion='capture_electronics host graph construction ONLY',
        original_measured_window_retained=True,first_three_real_eager_fields_retained=True,
        both_real_capture_field_outputs_retained=True,all_other_v2_nodes_retained=True,
        baseline_scope_unchanged=True,physical_optical_hardware_measured=False))
    return core.parallel_worker(args)


@contextmanager
def configured():
    entry=core.ENTRY;sources=core.source_map
    core.ENTRY=ENTRY;core.source_map=source_map
    try:
        with core.configured():
            core.original.worker=deployed_worker
            yield
    finally:core.ENTRY=entry;core.source_map=sources


if __name__=='__main__':
    with configured():core.original.main()
