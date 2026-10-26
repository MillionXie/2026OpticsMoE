"""Validate an explicitly measured causal DAG; never pad it to match a wall.

The future raw-event adapter must supply every host dispatch, synchronization,
stream/buffer dependency and witnessed scheduling interval. This validator does
not infer missing waits or invent latency from a final wall residual. It checks
every node start, not merely the final longest-path value.
"""
import math
from lgvq_async_trace_contract_20261026 import critical_path


def measured_dag(observations, *, measured_wall_ms, clock_tolerance_ms):
    if not 0 <= clock_tolerance_ms <= .25:
        raise ValueError('Explicit sub-millisecond clock uncertainty bound required')
    nodes = []; resources = {}
    for row in observations:
        a, b = row['observed_start_ms'], row['observed_end_ms']
        if not math.isfinite(a+b) or a < 0 or b < a:
            raise ValueError('Invalid original observed node interval')
        if not row.get('witness'):
            raise ValueError('Each interval must cite original host/CUDA evidence')
        nodes.append({k: v for k, v in row.items() if k not in ('observed_start_ms', 'observed_end_ms')} | dict(duration_ms=b-a))
        resource = row.get('resource')
        if resource and b > a:
            resources.setdefault(resource, []).append((a, b, row['id']))
    for name, intervals in resources.items():
        intervals.sort()
        for before, after in zip(intervals, intervals[1:]):
            if after[0] < before[1]-clock_tolerance_ms:
                raise ValueError('Impossible concurrent operations on serial resource '+name)
    computed = critical_path(nodes)
    for row in observations:
        if not math.isclose(computed['starts_ms'][row['id']], row['observed_start_ms'],
                            rel_tol=0, abs_tol=clock_tolerance_ms):
            raise ValueError('Missing real dependency/scheduling interval at '+row['id'])
    if not math.isclose(computed['wall_ms'], measured_wall_ms, rel_tol=0, abs_tol=clock_tolerance_ms):
        raise ValueError('Measured wall differs; do not append arbitrary padding')
    return dict(nodes=nodes, schedule=computed, every_original_node_start_reproduced=True,
        arbitrary_wall_padding_added=False, clock_tolerance_ms=clock_tolerance_ms,
        raw_event_adapter_still_required=True)
