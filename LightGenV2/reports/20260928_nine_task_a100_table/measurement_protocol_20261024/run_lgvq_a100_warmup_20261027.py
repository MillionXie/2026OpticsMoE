"""Explicit one-extra-real-field warmed diagnostic; NOT a cold-scope result."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import run_lgvq_a100_patch_parallel_20261027 as core
import lgvq_async_worker_20261026 as native
import lgvq_a100_warmup_20261027 as warm

ENTRY=Path(__file__).resolve();CORE_ENTRY=core.ENTRY;SOURCES=core.source_map


def source_map():
    result=SOURCES()
    for path in (CORE_ENTRY,ENTRY,ENTRY.with_name('lgvq_a100_warmup_20261027.py')):
        result[path.relative_to(core.original.REPO).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def accepted_report(report):
    return (report.get('status')=='complete' and report.get('timing_profile')==warm.PROFILE
        and report.get('qualified_for_final_hybrid_table') is False and report.get('original_cold_scope_qualified') is False
        and report.get('video_calls')==2048 and report.get('fields')==128
        and report.get('extra_pre_window_real_fields')==1 and report.get('all_predictions_bitwise_equal') is True
        and report.get('every_observed_node_start_reproduced') is True
        and report.get('physical_hardware_time_measured') is False
        and report.get('fixed_optics_projection',{}).get('status')=='COUNTERFACTUAL_FIXED_OPTICS_PROJECTION')


def warmed_worker(args):
    previous=native.worker;ns=dict(previous.__globals__,_prewarm=warm.prewarm,_save_warm=warm.save)
    exec(compile(warm.worker_AST(inspect.getsource(previous)),str(native.__file__)+'::extra_warmup','exec'),ns)
    native.worker=ns['worker']
    try:return core.parallel_worker(args)
    finally:native.worker=previous


@contextmanager
def configured():
    saved_entry=core.ENTRY;saved_sources=core.source_map;saved_accept=core.original.accepted_report
    core.ENTRY=ENTRY;core.source_map=source_map
    try:
        with core.configured():
            core.original.worker=warmed_worker;core.original.accepted_report=accepted_report
            yield
    finally:
        core.ENTRY=saved_entry;core.source_map=saved_sources;core.original.accepted_report=saved_accept


if __name__=='__main__':
    with configured():core.original.main()
