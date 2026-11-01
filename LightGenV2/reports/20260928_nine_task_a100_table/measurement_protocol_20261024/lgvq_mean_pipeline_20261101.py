"""User-approved mean-stage ideal pipeline, never a measured elapsed time.

CPU work is ideally balanced over 16 workers. Each of 32 fields receives the
same mean CPU service and mean device-stage durations. No last-job imbalance is
charged. Backend contention/dependencies remain explicit, so P+Q/N is used only
when the mean-stage schedule actually supports it. FFT kernels are replaced by
the specified 1886 Hz optical cycle; OS/off-CPU waits are not charged.
"""
import math
from collections import defaultdict
import lgvq_compute_scope_20261029 as scope
import recalculate_lg_front1886_20261030 as optical

CPU_PHASES = ('decode', 'RGB', 'crop', 'resize', 'normalization', 'patch')


def estimate(report, rows, tokens, variant, fields=32):
    if variant not in range(4) or len(rows) != fields * 16 or len(tokens) != fields:
        raise ValueError('Four cumulative variants, 16 videos per field required')
    selected = CPU_PHASES[3:] if variant == 0 else CPU_PHASES[:variant] + CPU_PHASES[3:]
    cpu = {p: sum(r.get(p + '_cpu_s', 0.) for r in rows) / (fields * 16) for p in selected}
    means = defaultdict(list)
    for row in report['raw_phases']:
        p = row['phase']
        means[p].append(optical.optical_s(1886) if p.startswith('optics') else row['kernel_union_s'])
    means = {p: sum(v) / len(v) for p, v in means.items()}
    means['tokenizer'] = sum(r['thread_cpu_s'] for r in tokens) / fields
    if any(not math.isfinite(v) or v < 0 for v in (*cpu.values(), *means.values())):
        raise ValueError('Finite nonnegative component means required')
    nodes = []
    for f in range(fields):
        previous = f'f{f-1}.CPU.patch' if f else None
        for p in selected:
            key = f'f{f}.CPU.{p}'
            nodes.append(dict(id=key, field=f, sequence=0, phase=p, resource='ideal_balanced_CPU16',
                              parents=[previous] if previous else [], duration_s=cpu[p]))
            previous = key
        inputs = previous
        previous = f'f{f}.tokenizer'
        nodes.append(dict(id=previous, field=f, sequence=0, phase='tokenizer', resource='cpu_tokenizer',
                          parents=[f'f{f-2}.decoder'] if f >= 2 else [], duration_s=means['tokenizer']))
        stages = ['emb_text', 'emb_conv3d', 'emb_position', 'adapter_visual_pool', 'adapter_visual']
        for i in range(1, 7):
            stages += [f'optics{i}', f'oeo{i}']
            if i == 3:
                stages += ['merger', 'adapter_language']
        stages += ['decoder']
        for seq, p in enumerate(stages, 1):
            key = f'f{f}.{p}'
            nodes.append(dict(id=key, field=f, sequence=seq, phase=p,
                              resource='optical' if p.startswith('optics') else 'GPU',
                              parents=[previous] + ([inputs] if p == 'emb_conv3d' else []),
                              duration_s=means[p]))
            previous = key
    schedule = scope.compute_schedule(nodes)
    P = sum(cpu.values())
    tail = schedule['total_s'] - fields * P
    after_image = [p for p in stages if p != 'emb_text']
    Q = sum(means[p] for p in after_image)
    closes = math.isclose(tail, Q, abs_tol=1e-9)
    return dict(status='MEAN_STAGE_IDEAL_PIPELINE_ESTIMATE_NOT_MEASURED_E2E', variant=variant,
                fields=fields, workers=16, CPU_phase_mean_s_per_field=cpu,
                backend_phase_mean_s_per_field=means, CPU_mean_s_per_field=P,
                mean_backend_after_image_s=Q, unhidden_tail_s=tail,
                tail_amortized_s_per_field=tail / fields,
                time16_s=schedule['total_s'] / fields,
                simple_P_plus_Q_over_N_verified=closes,
                load_imbalance_charged=False, CPU_dispatch_or_off_CPU_wait_charged=False,
                SLM_hz=1886, physical_optics_measured=False,
                text_work_recorded_not_zeroed=True, schedule=schedule)
