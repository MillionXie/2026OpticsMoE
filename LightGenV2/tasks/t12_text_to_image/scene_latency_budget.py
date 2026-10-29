"""Conservative additional-overhead budget, not physical optical timing.

Keep ALL measured software FFT/residual/host work and serially add the supplied
optical-cycle model. This is a serial-superset sensitivity bound, not the final
replacement DAG. Unknown driver/DMA/contention overhead H is never assumed zero.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timing',type=Path,required=True)
    p.add_argument('--baseline-key',default='electronic30');p.add_argument('--student-key',default='LG12')
    p.add_argument('--cycles',type=int,choices=[6,8,10],default=6);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and json.loads((a.timing/'report.json').read_text())['complete']
    source=a.timing/'summary.json';summary=json.loads(source.read_text());base=summary[a.baseline_key];student=summary[a.student_key]
    assert base['independent_processes']==student['independent_processes']==3 and base['n']==student['n']==24
    b,l=base['mean_ms'],student['mean_ms'];assert b>0 and l>0
    optical={6:6.269914,8:8.359886,10:10.449857}[a.cycles]
    report=dict(complete=True,software_measured_mean_ms=dict(baseline=b,student=l),software_speedup=b/l,
        optical_model_cycles=a.cycles,optical_cycle_model_ms=optical,
        serial_superset_formula='T_bound(H) = measured_software_LG + optical_cycle_model + H; retained-node costs must not exceed measured costs',
        extra_critical_path_overhead_budget_for_2x_ms=b/2-l-optical,
        additional_overhead_H='UNMEASURED: all extra driver, DMA, synchronization, and induced resource-contention delay',
        FFT_and_electronic_residual_still_fully_charged=True,
        user_zero_residual_cost_cannot_increase_this_superset_bound=True,
        exact_physical_replacement_DAG_not_completed=True,physical_optical_measurement=False,
        unknown_overhead_not_assumed_zero=True,joint_quality_and_speed_accepted=False,
        input_timing_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.mkdir(parents=True);(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
