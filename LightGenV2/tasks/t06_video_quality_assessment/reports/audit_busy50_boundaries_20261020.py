"""Offline reconciliation of independently measured busy time and pure costs.

This does not redefine a formal timing boundary or create missing service energy.
Waiting is real, but was explicitly excluded from the historical pure-cost table.
"""
from __future__ import annotations
import csv
import json
import statistics
from . import precision_comparison_20261019 as p


def reconcile(row):
    folder=(p.REPO/row['cost_source']).parent
    timing=p.rows(folder/'batch_timing.csv');telemetry=p.rows(folder/'telemetry.csv')
    waitsfile=folder/'excluded_waits.json'
    if not waitsfile.exists():waitsfile=folder/'excluded_pacing_waits.json'
    waits=p.read(waitsfile)
    cost=[t for t in timing if t['in_cost_subset']=='True']
    wait=[w for w in waits if w['in_cost_subset']]
    assert len(cost)==len(wait) and sum(int(t['videos']) for t in cost)==200
    def bounds(w):return w.get('start',w.get('start_monotonic_s')),w.get('end',w.get('end_monotonic_s'))
    def duration(w):a,b=bounds(w);return b-a
    pure=sum(float(t['synchronized_wall_ms']) for t in cost)*16/200
    actual_wait=sum(map(duration,wait))*1000*16/200
    p.close(pure,row['time_ms'])
    p.close(actual_wait,row['excluded_wait_16_ms'])
    active=[t for t in telemetry if t['phase'].startswith('active:')]
    active_busy=statistics.mean(float(t['utilization_percent']) for t in active)
    # Show the actual recurrent schedule, not a single-job end-to-end latency.
    # Include every imposed wait for the normalized steady-state throughput audit;
    # omit the final wait separately when considering a finite200-video interval.
    requested=sum(w['requested_s'] for w in wait)*1000*16/200
    with_wait=pure+actual_wait
    finite_without_terminal_wait=pure+sum(duration(w) for w in wait[:-1])*80
    return dict(model=row['model'],precision=row['precision'],batch=row['batch'],
        independent_gpu_busy_percent=row['utilization_percent'],
        cost_active_tagged_driver_busy_percent_audit_only=active_busy,
        pure_inference_16_ms=pure,excluded_requested_wait_16_ms=requested,
        excluded_actual_wait_16_ms=actual_wait,
        inference_plus_all_cycle_waits_16_ms_audit_only=with_wait,
        inference_plus_internal_waits_excluding_terminal_16_ms_audit_only=finite_without_terminal_wait,
        wait_inclusive_to_pure_ratio=with_wait/pure,
        pure_energy_16_j=row['energy_j'],
        true_service_energy_16_j=None,
        note='Wait-inclusive audit still excludes decode, preprocessing, H2D and scheduling gaps; not measured end-to-end service latency. Independent busy labels are NOT the utilization of this cost run. Service energy cannot be reconstructed by multiplying pure active power by wait-inclusive time.',
        cost_source=row['cost_source'],utilization_source=row['utilization_source'])


def main():
    data=p.read(p.RENTAL/'presentation/precision_comparison.json');rows=[]
    for values in data['groups'].values():
        for row in values:
            if row['condition']=='Approximately 50%' and row.get('plotted',True):rows.append(reconcile(row))
    result=dict(status='audited',formal_costs_unchanged=True,
        conclusion='Pure per-call inference costs exclude the operation that causes lower busy time; these figures cannot establish the total time/energy penalty of50% temporal throttling.',
        figures_should_not_claim_matched_busy_cost_window=True,
        control='Frozen real-work/idle duty after calibration; no fake workload, SM restriction, memory partition or power change.',rows=rows)
    out=p.RENTAL/'presentation'
    (out/'busy50_boundary_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'busy50_boundary_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
