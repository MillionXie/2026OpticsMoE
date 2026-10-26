"""Fixed-component energy proxy on a projected DAG, never NVML FFT energy.

Apply the already approved component powers to interval UNIONS. The controller
is counted once across overlapping optics/electronics. Serial execution reduces
exactly to 121.746*t_optical + 104.230*t_electronic (seconds).
"""
import math

CONTROLLER_W = 41.388
OPTICAL_DEVICES_W = 80.358
ELECTRONIC_BOARD_W = 62.842
EXCLUDED = {'detector_ROI', 'electronic_residual', 'capture_excluded', 'H2D'}
ELECTRONIC = {'tokenizer', 'frontend', 'electronics', 'dispatch', 'sync', 'capture_electronics'}


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not math.isfinite(start+end) or start < 0 or end < start:
            raise ValueError('Finite nonnegative ordered intervals required')
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def length(intervals):
    return sum(end-start for start, end in interval_union(intervals))


def component_energy(nodes, schedule, fields):
    if not isinstance(fields, int) or fields < 1:
        raise ValueError('Positive complete field count required')
    if set(schedule['starts_ms']) != {n['id'] for n in nodes} or set(schedule['ends_ms']) != {n['id'] for n in nodes}:
        raise ValueError('Complete DAG schedule required')
    optical = []; electronic = []
    for node in nodes:
        a, b = schedule['starts_ms'][node['id']], schedule['ends_ms'][node['id']]
        if node['kind'] in EXCLUDED:
            if b != a:
                raise ValueError('Excluded operation must already be zero in the projected DAG')
        elif node['kind'] == 'optical_simulation':
            optical.append((a, b))
        elif node['kind'] in ELECTRONIC:
            electronic.append((a, b))
        else:
            raise ValueError('Unknown component scope; do not silently charge or exclude it')
    optical_ms, electronic_ms = length(optical), length(electronic)
    if not math.isclose(optical_ms, sum(b-a for a, b in optical), rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError('This worker has one serial optical/body resource, not concurrent optical fields')
    controller_ms = length(optical+electronic)
    if not math.isclose(controller_ms, schedule['wall_ms'], rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError('Unclassified projected wall gap; no energy may silently disappear')
    controller_j = CONTROLLER_W*controller_ms/1000
    optical_j = OPTICAL_DEVICES_W*optical_ms/1000
    electronic_j = ELECTRONIC_BOARD_W*electronic_ms/1000
    total = controller_j+optical_j+electronic_j
    return dict(status='FIXED_COMPONENT_UNION_PROXY_NOT_METERED', energy_j=total,
        energy16_j=total/fields, controller_j=controller_j, optical_devices_j=optical_j,
        electronic_board_j=electronic_j, controller_union_ms=controller_ms,
        optical_union_ms=optical_ms, electronic_union_ms=electronic_ms,
        optical_electronic_overlap_ms=optical_ms+electronic_ms-controller_ms,
        powers_w=dict(controller=CONTROLLER_W, optical_devices=OPTICAL_DEVICES_W,
                      electronic_board=ELECTRONIC_BOARD_W),
        controller_counted_once=True, board_counted_once_across_parallel_streams=True,
        baseline_host_proxy_used=False, software_FFT_NVML_energy_substituted=False,
        physical_optical_energy_measured=False)
