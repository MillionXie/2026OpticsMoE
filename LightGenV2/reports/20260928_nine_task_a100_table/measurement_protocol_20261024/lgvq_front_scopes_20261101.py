"""Fresh four cumulative frontend scopes with native pixels and real backend.

V0 starts with native RGB crops. V1..V3 execute the full original file decoder
online; their ideal compute estimates charge decode, decode+RGB, or all three.
Real elapsed windows retain all operations (including excluded ones), FFmpeg
decode is explicitly single-threaded to avoid hidden >16-worker CPU capacity.
"""
import ast
from contextlib import contextmanager
import inspect
import math
import textwrap
import threading
import time
from pathlib import Path

import lgvq_compute_scope_20261029 as scope
import lgvq_dynamic_cpu_20261101 as dynamic
import lgvq_resize_norm_20261029 as prep
import lgvq_patch_layout_20261029 as layout
import lgvq_mean_pipeline_20261101 as mean

VARIANT = None
LOCAL = threading.local()
DECODER = None


@contextmanager
def segment(phase):
    a = time.perf_counter(); ca = time.thread_time()
    try:
        yield
    finally:
        cb = time.thread_time(); b = time.perf_counter()
        LOCAL.measurements.append(dict(phase=phase, host_start=a, host_end=b,
                                       thread_cpu_s=cb-ca, elapsed_s=b-a,
                                       off_CPU_or_clock_difference_s=(b-a)-(cb-ca)))


def observed_decoder(original):
    """Instrument native statement boundaries, preserving reads, crops, resize.

The only capture-policy change is explicit single-thread FFmpeg. Strict native
resize/normalization/BF16/output byte comparisons are still mandatory.
"""
    source = textwrap.dedent(inspect.getsource(original)); tree = ast.parse(source)
    fn = tree.body[0]
    counts = dict(decode=0, RGB=0, crop=0, resize=0)

    def wrap(node, phase):
        counts[phase] += 1
        return ast.With(items=[ast.withitem(context_expr=ast.Call(
            func=ast.Name(id='_segment', ctx=ast.Load()), args=[ast.Constant(phase)], keywords=[]))], body=[node])

    class Hook(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            target = ast.unparse(node.targets[0])
            if target == 'capture':
                node.value = ast.parse('_capture(str(path))', mode='eval').body
                return wrap(node, 'decode')
            if target in ('path', 'count', 'ok, bgr', '(ok, bgr)'):
                return wrap(node, 'decode')
            if target == 'rgb':
                return wrap(node, 'RGB')
            if target in ('height, width', '(height, width)', 'side', 'top, left', '(top, left)'):
                return wrap(node, 'crop')
            if target == 'square':
                crop = node.value.args[0]
                cropped = ast.Assign(targets=[ast.Name(id='_native_crop', ctx=ast.Store())], value=crop)
                node.value.args[0] = ast.Name(id='_native_crop', ctx=ast.Load())
                return [wrap(cropped, 'crop'), wrap(node, 'resize')]
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Call):
                name = ast.unparse(node.value.func)
                if name in ('capture.set', 'capture.release'):
                    return wrap(node, 'decode')
                if name == 'frames.append':
                    return wrap(node, 'resize')
            return node

    tree = ast.fix_missing_locations(Hook().visit(tree))
    if counts != dict(decode=7, RGB=1, crop=4, resize=2):
        raise ValueError('Native decode AST changed: ' + repr(counts))

    def capture(path):
        import cv2
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if int(cap.get(cv2.CAP_PROP_N_THREADS)) != 1:
            cap.release(); raise ValueError('One actual FFmpeg decoder thread required')
        return cap

    ns = dict(original.__globals__, _segment=segment, _capture=capture)
    exec(compile(tree, '<timed_native_decode_RGB_crop>', 'exec'), ns)
    return ns[fn.name], dict(native_decode_source_sha256=scope.digest(source.encode()),
                             observed_AST_sha256=scope.digest(ast.dump(tree).encode()),
                             crop_is_native_slice_no_extra_copy=True,
                             FFmpeg_threads_per_CPU_worker=1, instrumented_statements=counts)


def configure(variant):
    global VARIANT, DECODER, DECODER_RECEIPT
    if variant not in range(4):
        raise ValueError('Cumulative scope must be 0..3')
    VARIANT = variant
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
    DECODER, DECODER_RECEIPT = observed_decoder(native.decode_frames)


def patch_video(prepared, field, slot, patch_function):
    if VARIANT == 0:
        pixels, row = dynamic.patch_video(prepared, field, slot, patch_function)
        row.update(variant=0, decode_cpu_s=0., RGB_cpu_s=0., crop_cpu_s=0.,
                   added_frontend_executed_online=False, frontend_measurements=[])
        return pixels, row
    proc = prep.processor(prepared['processor_model_path'])
    LOCAL.measurements = []; prep.LOCAL.processor_resize_cpu_s = 0.
    begin = time.perf_counter(); ca = time.thread_time()
    images = DECODER(Path(prepared['video_path']), 4)
    # Native decoder already performs resize/PIL; never repeat it here.
    resized = [__import__('numpy').asarray(image) for image in images]
    resize_end = time.perf_counter()
    cb = time.thread_time()
    normalized = proc(images=images, return_tensors='pt')
    cc = time.thread_time(); norm_end = time.perf_counter()
    if len(normalized) != 1:
        raise ValueError('Native four-frame geometry required')
    value, h, w, t, p, m = normalized[0]
    pixels = layout.strided_patch(value, h, w, t, p, m)
    cd = time.thread_time(); end = time.perf_counter()
    parts = {phase: sum(r['thread_cpu_s'] for r in LOCAL.measurements if r['phase'] == phase)
             for phase in ('decode', 'RGB', 'crop', 'resize')}
    extra = prep.LOCAL.processor_resize_cpu_s
    parts['resize'] += extra
    parts['normalization'] = cc-cb-extra
    parts['patch'] = cd-cc
    if any(not math.isfinite(v) or v < 0 for v in parts.values()):
        raise ValueError('Bad active CPU partition')
    selected = ('decode', 'RGB', 'crop')[:VARIANT] + ('resize', 'normalization', 'patch')
    row = dict(field=field, slot=slot, sample_id=prepared['sample_id'], variant=VARIANT,
               worker_thread=threading.get_ident(), worker_native_id=threading.get_native_id(),
               host_start=begin, host_end=end, resize_end_wall=resize_end,
               normalization_end_wall=norm_end, actual_thread_cpu_s=cd-ca,
               thread_cpu_s=sum(parts[p] for p in selected),
               unassigned_instrumentation_CPU_s=(cd-ca)-sum(parts.values()),
               processor_resize_cpu_s=extra, frontend_measurements=list(LOCAL.measurements),
               frontend_decoder=DECODER_RECEIPT, added_frontend_executed_online=True,
               actual_wall_includes_all_three_even_when_scope_excludes_some=True,
               raw_patch_bytes_equal=False, validation_deferred_until_after_formal_window=True,
               CPU_dispatch_waits_retained_in_wall_not_charged_in_projection=True,
               **{p+'_cpu_s': v for p, v in parts.items()})
    dynamic.PENDING.append((pixels, resized, value, prepared, row))
    return pixels, row


def prepare_video(row):
    result = dynamic.prepare_video(row)
    result['video_path'] = row['video_path']
    return result


def start(pool, prepared, patch_fn):
    if dynamic.FUTURES:
        raise ValueError('Fresh worker required')
    for i, p in enumerate(prepared):
        dynamic.FUTURES.append(pool.submit(patch_video, p, i//16, i%16, patch_fn))


futures = dynamic.futures
validate = dynamic.validate


def summarize_compute(kernels, rows, tokens, fields):
    # This older builder is used only to preserve raw device phase extraction.
    # Its historical-thread schedule is diagnostic, not the headline estimate.
    report = prep.NATIVE_SUMMARY(kernels, rows, tokens, fields)
    result = mean.estimate(report, rows, tokens, VARIANT, fields)
    by = {n['id']: n for n in result['schedule']['nodes']}
    contributions = {}
    for key in result['schedule']['additive_path']:
        n = by[key]
        contributions[n['phase']] = contributions.get(n['phase'], 0.) + n['duration_s']/fields
    report.update(profile='LG_FOUR_CUMULATIVE_MEAN_SCOPE_20261101', variant=VARIANT,
                  legacy_raw_work_schedule_s=report['time16_s'],
                  schedule=result['schedule'], time16_s=result['time16_s'],
                  equivalent_s_per_group=contributions,
                  mean_stage_pipeline=result, CPU_preprocessing_raw_parts=rows,
                  CPU_resize_and_normalization_included=True, SLM_hz=1886,
                  optical_layer_s=1/1886+.000300+.0000307,
                  CPU_file_read_decode_RGB_crop_excluded=VARIANT == 0,
                  CPU_quality_pixel_input_preparation_excluded=True,
                  actual_FIFO_worker_queue=True, ideal_mean_estimate_no_load_imbalance=True,
                  real_CPU_wall_s=max(r['host_end'] for r in rows)-min(r['host_start'] for r in rows),
                  real_CPU_wall_is_not_compute_only_projection=True)
    for row in report['raw_phases']:
        if row['phase'].startswith('optics'):
            row['charged_compute_s'] = report['optical_layer_s']
    return report
