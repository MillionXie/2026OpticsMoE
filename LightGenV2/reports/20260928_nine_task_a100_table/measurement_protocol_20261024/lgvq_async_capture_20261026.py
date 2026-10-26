"""Two independent real-field captures for the asynchronous LG candidate.

This is a worker component, not a qualified benchmark. Callers own the first
three eager fields, online tokenizer, guard, measured continuous clock, power
trace, complete score validation and dependency-aware optical projection.
No Torch/model import or GPU work happens merely by importing this module.
"""
import ast
import inspect
import time


def scoring_partition(model, score_module, partition_module):
    """Partition the approved score AST, never reintroduce auxiliary losses."""
    original = type(model).forward
    score_tree, score_receipt = score_module.score_forward_AST(inspect.getsource(original))
    tree, partition_receipt = partition_module.partition_forward(ast.unparse(score_tree))
    namespace = dict(original.__globals__)
    # Use the approved scoring-only router; routing arithmetic is unchanged.
    router_tree, router_receipt = score_module.router_score_AST(
        inspect.getsource(namespace['_standardized_router']))
    exec(compile(router_tree, inspect.getsourcefile(original)+'::async_router', 'exec'), namespace)
    exec(compile(tree, inspect.getsourcefile(original)+'::async_partition', 'exec'), namespace)
    return namespace['prepare_native_prefix'], namespace['forward_from_prefix'], dict(
        scoring=score_receipt, partition=partition_receipt, router=router_receipt,
        original_scoring_tail_AST_unchanged=True, auxiliary_outputs_omitted=True)


def serial_capture_plan(fields=128):
    if fields <= 5:
        raise ValueError('Full real-field workload required')
    return ['eager_real']*3 + ['capture_slot0_real', 'capture_slot1_real'] + ['async_replay_real']*(fields-5)


def capture_real_slot(torch, *, field, raw, cpu_ids, cpu_mask, model,
                      front_stream, body_stream, prepare_prefix, forward_tail,
                      tape_factory, install_boundaries):
    """Capture/execute one REAL field exactly once; keep every startup cost.

    Frontend capture is replayed once for this field, before capturing the
    original body. This provides actual current prefix values without a dummy
    inference. Both capture phases are globally quiescent; all synchronization
    and construction stays within the caller's continuous measured clock.
    Independent default graph pools and fresh static inputs are mandatory.
    """
    started = time.perf_counter()
    initial_sync_start = time.perf_counter()
    torch.cuda.synchronize()
    initial_sync_end = time.perf_counter()
    static = {name: torch.empty_like(raw[name]) for name in ('pixels', 'quality', 'ids', 'mask')}
    ids = cpu_ids.pin_memory()
    mask = cpu_mask.bool().pin_memory()
    with torch.cuda.stream(front_stream):
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        h2d_host_start = time.perf_counter()
        h0.record()
        static['ids'].copy_(ids, non_blocking=True)
        static['mask'].copy_(mask, non_blocking=True)
        h1.record()
        h2d_host_end = time.perf_counter()
        d0 = torch.cuda.Event(enable_timing=True); d1 = torch.cuda.Event(enable_timing=True)
        refresh_host_start = time.perf_counter(); d0.record()
        for name in ('pixels', 'quality'):
            static[name].copy_(raw[name], non_blocking=True)
        d1.record(); refresh_host_end = time.perf_counter()
    # capture_begin(global) must not run with outstanding work on another stream.
    transfer_sync_start = time.perf_counter(); torch.cuda.synchronize(); transfer_sync_end = time.perf_counter()
    tape = tape_factory(torch, front_stream)
    events = [dict(kind='frontend', label='input_device_refresh', start=d0, end=d1,
                   stream='frontend', host_submit_start=refresh_host_start, host_submit_end=refresh_host_end)]
    captures = []
    restored = None
    try:
        tape.start('electronics', 'original_frontend_and_input_adapters')
        prefix = prepare_prefix(static)
        tape.close()
        if len(tape.segments) != 1:
            raise RuntimeError('Frontend must be one independent graph')
        tape.segments[0]['kind'] = 'frontend'
        with torch.cuda.stream(front_stream):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            submit_start = time.perf_counter()
            a.record(); tape.segments[0]['graph'].replay(); b.record()
            events.append(dict(kind='frontend', label='original_frontend_and_input_adapters',
                               start=a, end=b, stream='frontend',
                               host_submit_start=submit_start, host_submit_end=time.perf_counter()))
        front_sync_start = time.perf_counter(); b.synchronize(); front_sync_end = time.perf_counter()
        captures.append(dict(phase='frontend', start=started, end=time.perf_counter()))
        tape.stream = body_stream
        restored, ledger = install_boundaries(model, tape)
        ca = time.perf_counter()
        tape.capturing = True
        tape.start('electronics', 'original_body_before_first_optics')
        result = forward_tail(prefix)
        tape.close()
        tape.capturing = False
        captures.append(dict(phase='body', start=ca, end=time.perf_counter()))
        kinds = [s['kind'] for s in tape.segments]
        if (len(kinds) != 34 or kinds.count('frontend') != 1
                or kinds.count('electronics') != 17 or kinds.count('optical_simulation') != 6
                or kinds.count('detector_ROI') != 6 or kinds.count('electronic_residual') != 4):
            raise RuntimeError('Original complete six-pass body boundaries changed')
        with torch.cuda.stream(body_stream):
            body_stream.wait_event(b)
            for segment in tape.segments[1:]:
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                submit_start = time.perf_counter()
                a.record(); segment['graph'].replay(); b.record()
                events.append(dict(kind=segment['kind'], label=segment['label'],
                                   start=a, end=b, stream='body',
                                   host_submit_start=submit_start, host_submit_end=time.perf_counter()))
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            submit_start = time.perf_counter(); a.record(); prediction = result['prediction'].clone(); b.record()
            events.append(dict(kind='electronics', label='retain_current_score', start=a, end=b, stream='body',
                               host_submit_start=submit_start, host_submit_end=time.perf_counter()))
        score_sync_start = time.perf_counter(); b.synchronize(); score_sync_end = time.perf_counter()
        prefix_owned = {k: v for k, v in prefix.items() if torch.is_tensor(v)}
        if not prefix_owned:
            raise RuntimeError('Prefix must retain its real captured tensor allocations')
        slot = dict(static=static, prefix_owned=prefix_owned, prefix=prefix,
                    segments=tape.segments, prediction=result['prediction'],
                    score_ready=b, keepalive=(ids, mask), tape=tape)
        record = dict(field=field, captures=captures, events=events,
                      graph_construction=[{k: v for k, v in s.items() if k != 'graph'} for s in tape.segments],
                      original_H2D_events=(h0, h1), score_ready=b,
                      H2D_host_submit_start=h2d_host_start, H2D_host_submit_end=h2d_host_end,
                      synchronization_observations=[
                          dict(kind='device_before_buffers', start=initial_sync_start, end=initial_sync_end),
                          dict(kind='device_after_transfer', start=transfer_sync_start, end=transfer_sync_end),
                          dict(kind='frontend_before_body_capture', start=front_sync_start, end=front_sync_end),
                          dict(kind='score_ready', start=score_sync_start, end=score_sync_end)],
                      host_start=started, host_end=time.perf_counter(),
                      no_extra_or_discarded_forward=True, optical_replacement_ledger=ledger)
        return slot, prediction, record
    finally:
        tape.capturing = False
        try:
            if tape.context is not None:
                tape.close()
        finally:
            if restored is not None:
                restored()
