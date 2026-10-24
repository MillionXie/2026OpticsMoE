"""One provenance index for both tables; missing new measurements stay missing.

This is an intermediate, machine-readable index, not a completed paper table.
The same immutable Qwen100 anchor must serve both tables. No historical narrow
cost is a fallback for missing input-inclusive measurements or OP counts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
from audit_t07 import audit,audit_qwen,read,sha
from audit_salicon import audit as audit_salicon


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous index and source receipts')
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as t06
    manifest=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
    paths=[t06.CANONICAL/'qwen100_repaired01/qwen_cap100_repeat01',
           t06.CANONICAL/'qwen100_remaining01/qwen_cap100_repeat02',
           t06.CANONICAL/'qwen100_remaining01/qwen_cap100_repeat03']
    references=[dict(path=str(p.relative_to(REPO)),**t06.audit_point(p,manifest)) for p in paths]
    originals=[read(path/'report.json') for path in paths]
    tokens=[r['effective_input_tokens200'] for r in originals]
    if tokens!=[89200]*3:raise ValueError('Shared Qwen token work changed')
    seconds=sum(r['cost']['continuous_wall_s'] for r in references)
    energy=sum(r['cost']['estimated_system_energy_j'] for r in references)
    anchor=dict(status='independently_audited_600_calls',calls=600,unique_videos=200,frames_per_video=4,
        sample_ms=1000*seconds/600,equivalent_frame_ms=1000*seconds/2400,sample_j=energy/600,
        equivalent_frame_j=energy/2400,time_weighted_board_w=sum(r['cost']['gpu_energy_j'] for r in references)/seconds,
        common_input_tokens_total=sum(tokens),common_tokens_per_s=sum(tokens)/seconds,
        full558_reference_srcc=originals[0]['original_full558_reference']['srcc'],
        measured200_subset_srcc=[r['performance200']['srcc'] for r in originals],
        raw_references=references,scope='file to CPU score, continuous whole window, all199 gaps, no dedicated inference warmup')
    protocol=read(HERE/'protocol.json')
    rows={item['id']:dict(id=item['id'],task=item['task'],metric=item['metric'],
                         full_test_reference=dict(ours=item['ours_target'],qwen=item['qwen_target']),
                         ours=None,qwen=None,ops_status='pending actual matching-input untimed replay',
                         status='not_yet_complete_under_new_protocol') for item in protocol['rows']}
    rows['01a']['qwen']=anchor
    t07=REPO/'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation'
    enrolled=t07/'abo200_enrolled_protocol_20260913/protocol.json'
    ours=audit(t07/'a100_input_inclusive_20261024_01',enrolled)
    qwen=audit_qwen(t07/'a100_input_inclusive_20261024_qwen01',enrolled)
    rows['03'].update(ours=ours,qwen=qwen,status='matched_three_round_timing_energy_independently_audited; OP pending')
    our_s=sum(r['total_ms']*200/1000 for r in ours['repeats']);qwen_s=sum(r['seconds'] for r in qwen['repeats'])
    common=qwen['aggregate']['common_input_tokens_total']
    rows['03']['common_tokens']=dict(total=common,ours_per_s=common/our_s,qwen_per_s=common/qwen_s,
                                    rate_ratio=qwen_s/our_s,time_ratio=qwen_s/our_s,basis='same original Qwen valid input IDs/masks for each query')
    sali=REPO/'LightGenV2/tasks/t03_saliency/runs/simulation'
    verified=audit_salicon(sali/'a100_input_inclusive_20261024_01',sali/'moe_alpha40_sam_batch8_crosssample_20260913_seed42/manifests/samples.csv')
    rows['06'].update(ours=verified['aggregate']['ours'],qwen=verified['aggregate']['qwen'],
                      status='matched_three_round_timing_energy_independently_audited; OP pending',raw_audit=verified)
    common=600*196
    our_s=verified['aggregate']['ours']['sample_ms']*.6;qwen_s=verified['aggregate']['qwen']['sample_ms']*.6
    rows['06']['common_tokens']=dict(total=common,ours_per_s=common/our_s,qwen_per_s=common/qwen_s,
                                    rate_ratio=qwen_s/our_s,time_ratio=qwen_s/our_s,basis='196 common native visual input patches/image; no language, not generated tokens')
    value=dict(protocol_id=protocol['protocol_id'],status='PARTIAL_NOT_FINAL_PAPER_TABLE',
        shared_qwen_temporal100=anchor,nine_metric_rows=list(rows.values()),
        reference_for_both_tables='shared_qwen_temporal100 in this exact index; never select another date or trial',
        current_input_protocol_sha256=sha(HERE/'protocol.json'),
        no_predefined_speedup_claim=True,derived_optical_vs_measured_electronic_boundaries_separate=True,
        raw_data_in_original_task_runs=True,power_no_idle_subtraction=True)
    with args.output.open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=value['status'],shared_qwen100={k:anchor[k] for k in ('calls','sample_ms','sample_j','time_weighted_board_w')},
                          complete_timing_energy_rows=['03','06'],pending_rows=['01a ours','01b','02','04','05','07','08'],all_operation_counts='pending')))


if __name__=='__main__':main()
