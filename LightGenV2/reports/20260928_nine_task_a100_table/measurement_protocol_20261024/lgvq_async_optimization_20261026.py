"""Optional fresh-run runtime policy and explicit electronic phase boundaries.

No inference, warmup, cached features, dtype changes or GPU actions on import.
EAGER module loading is selected before worker startup, never by a dummy forward.
The capture policy keeps synchronization and every capture but does not force a
full Python GC / allocator-cache purge. All remaining preparation stays timed.
"""
from contextlib import contextmanager
import time


def prepare_without_cache_purge(tape):
    if tape.preparation is not None:
        return
    started = time.perf_counter()
    tape.torch.cuda.synchronize()
    tape.preparation_calls += 1
    tape.preparation = dict(host_start=started, host_end=time.perf_counter(),
        host_ms=(time.perf_counter()-started)*1000,
        charged_in_first_electronics_segment=True, calls=tape.preparation_calls,
        extra_forward_calls=0, forced_gc_collect=False, allocator_cache_purge=False)
    tape.capture_audit['preparation'] = tape.preparation


def detailed_boundaries(model, tape, original_install):
    restore, ledger = original_install(model, tape)
    saved = []
    try:
        for name in ('frame_merger', 'readout'):
            owner = getattr(model, name)
            original = owner.forward
            def wrapped(*args, _original=original, _name=name, **kwargs):
                with tape.excluded(_name, 'electronics'):
                    return _original(*args, **kwargs)
            owner.forward = wrapped
            saved.append((owner, original))
            ledger.append(dict(stage=name, original_math_unchanged=True,
                               detailed_electronics_boundary=True))
    except BaseException:
        for owner, original in reversed(saved):
            owner.forward = original
        restore()
        raise
    def close():
        for owner, original in reversed(saved):
            owner.forward = original
        restore()
    return close, ledger


@contextmanager
def runtime_gc_policy(enabled):
    import gc
    was_enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        elif gc.isenabled():
            gc.disable()
