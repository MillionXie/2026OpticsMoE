"""One-change candidate: prepare capture once, retain the original 33 graphs.

The unchanged projection worker still executes its first three real fields,
captures all original operations and checks every prediction against its sealed
reference. Only Tape.start/close changes: use the public Torch 2.6
CUDAGraph.capture_begin/capture_end on the SAME explicit stream, instead of
repeating torch.cuda.graph's synchronize/gc.collect/empty_cache for every
segment. The one initial preparation remains inside the first charged
electronics capture interval. No extra forward, dummy warmup, omitted graph,
shared memory pool, asynchronous queue, or auxiliary-output change is added.

Version-pinned upstream API/source reviewed:
https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/graphs.py
capture_begin/end use the current stream; graph.__enter__ repeats the three
preparation operations. This candidate retains capture_error_mode='global'.
Independent default graph pools avoid introducing shared-pool lifetime rules.
The original strict guard and exact replay order are owned by the reused worker.
"""

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import weakref

import lgvq_physical_projection_20261026 as projection
ORIGINAL_WORKER = projection.worker


class LowLevelTape(projection.Tape):
    """Keep original observation scopes and replay ledger, change capture API."""

    instances = weakref.WeakSet()
    audits = []  # Scalar receipts only, never keeps CUDA tensors/graphs alive.

    def __init__(self, torch, stream):
        super().__init__(torch, stream)
        self.preparation = None
        self.preparation_calls = 0
        self.capture_audit = dict(preparation=None, segment_kinds=[])
        type(self).audits.append(self.capture_audit)
        type(self).instances.add(self)

    def start(self, kind, label):
        if self.context is not None:
            raise RuntimeError('Cannot nest or overwrite an active capture')
        started = time.perf_counter()
        if self.preparation is None:
            # These are deliberately timed as part of the first electronics
            # segment. They do not execute an additional inference.
            if kind != 'electronics':
                raise RuntimeError('Initial capture preparation must be charged')
            self.torch.cuda.synchronize()
            gc.collect()
            self.torch.cuda.empty_cache()
            prepared = time.perf_counter()
            self.preparation_calls += 1
            self.preparation = dict(
                host_start=started, host_end=prepared,
                host_ms=(prepared-started)*1000,
                charged_in_first_electronics_segment=True,
                calls=self.preparation_calls,
                extra_forward_calls=0,
            )
            self.capture_audit['preparation'] = self.preparation
        graph = self.torch.cuda.CUDAGraph()
        stream_context = self.torch.cuda.stream(self.stream)
        stream_context.__enter__()
        try:
            # No pool argument: preserve a distinct default pool per graph.
            graph.capture_begin(capture_error_mode='global')
        except BaseException:
            stream_context.__exit__(*sys.exc_info())
            raise
        self.context = (stream_context, graph, kind, label, started)

    def close(self):
        if self.context is None:
            raise RuntimeError('No active capture to close')
        stream_context, graph, kind, label, started = self.context
        try:
            graph.capture_end()
        finally:
            self.context = None
            stream_context.__exit__(None, None, None)
        ended = time.perf_counter()
        self.segments.append(dict(
            graph=graph, kind=kind, label=label,
            capture_host_start=started, capture_host_end=ended,
            capture_host_ms=(ended-started)*1000,
            capture_API='CUDAGraph.capture_begin/capture_end',
            capture_error_mode='global', independent_default_pool=True,
        ))
        self.capture_audit['segment_kinds'].append(kind)


def _sources():
    paths = (Path(__file__).resolve(), Path(projection.__file__).resolve())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def worker(args, guard, N):
    """Call the unmodified worker with only its Tape class temporarily replaced.

    Keep projection.__file__ unchanged: its existing module-deployment SHA gate,
    original worker source paths and output schema remain truthful. This extra
    identity freezes the candidate itself as well. The controller must publish
    and record both files; no source is copied into an already running worker.
    """
    if N != 2048:
        raise ValueError('This isolated candidate is fixed to 2048 video calls')
    frozen = _sources()
    before = set(LowLevelTape.instances)
    audit_start = len(LowLevelTape.audits)
    identity_path = args.output/'lowlevel_capture_identity.json'
    identity = dict(
        status='running', candidate='one_initial_capture_preparation_20261026',
        sources=frozen, video_calls=N,
        upstream='https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/graphs.py',
        original_projection_worker_reused=True, original_scopes_and_math_unchanged=True,
        original_first48_real_results_retained=True,
        capture_error_mode='global', shared_graph_pool=False,
        initial_preparation_charged=True, asynchronous_queue_added=False,
        auxiliary_outputs_changed=False, extra_forward_calls=0,
    )
    def save():
        identity_path.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
    save()
    original_tape = projection.Tape
    projection.Tape = LowLevelTape
    try:
        result = ORIGINAL_WORKER(args, guard, N)
        tapes = [t for t in LowLevelTape.instances if t not in before]
        for tape in tapes:
            if tape.context is not None or tape.preparation_calls != 1:
                raise RuntimeError('Incomplete capture or repeated initial preparation')
        audits = LowLevelTape.audits[audit_start:]
        if len(audits) != 1:
            raise RuntimeError('Exactly one original 33-segment capture is required')
        kinds = audits[0]['segment_kinds']
        if (len(kinds) != 33 or kinds.count('electronics') != 17
                or kinds.count('optical_simulation') != 6
                or kinds.count('detector_ROI') != 6
                or kinds.count('electronic_residual') != 4
                or audits[0]['preparation']['calls'] != 1):
            raise RuntimeError('Original capture scopes or single preparation changed')
        if _sources() != frozen:
            raise RuntimeError('Capture candidate/original source changed during run')
        identity.update(status='complete', sources_unchanged=True,
                        capture_audits=audits,
                        original_worker_full_output_validation_passed=True)
        save()
        return result
    except BaseException as error:
        # A failed capture must not leave the stream context active while the
        # outer controller/guard performs its normal stop-and-preserve cleanup.
        for tape in list(LowLevelTape.instances):
            if tape not in before and tape.context is not None:
                try:
                    tape.close()
                except BaseException:
                    pass
        identity.update(status='failed', error=repr(error),
                        sources_unchanged=(_sources() == frozen))
        save()
        raise
    finally:
        projection.Tape = original_tape
