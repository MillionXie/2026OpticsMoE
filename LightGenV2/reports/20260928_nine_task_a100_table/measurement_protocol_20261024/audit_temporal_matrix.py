"""Fixed canonical run map for both temporal tables; no 'latest'/best selection.

Only independently replayed complete triples get aggregate values. Pending
points stay missing. This CPU index is not the final paper Excel or OP count.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry


def paths(model,cap):
    root=entry.CANONICAL
    if model=='qwen' and cap==100:
        return [root/'qwen100_repaired01/qwen_cap100_repeat01',
                root/'qwen100_remaining01/qwen_cap100_repeat02',root/'qwen100_remaining01/qwen_cap100_repeat03']
    if model=='qwen':return [root/'lab_remaining01/qwen50'/f'qwen_cap50_repeat{i:02d}' for i in (1,2,3)]
    if model=='glm' and cap==100:
        return [root/'lab_remaining01/glm_clip_yolo'/f'glm_cap100_repeat{i:02d}' for i in (1,2)]+[
                root/'lab_remaining02/glm100_r3/glm_cap100_repeat03']
    sub='glm50' if model=='glm' else 'clip_yolo'
    return [root/'lab_remaining02'/sub/f'{model}_cap{cap}_repeat{i:02d}' for i in (1,2,3)]


def aggregate(reports,audits):
    if len(audits)!=3 or any(a['status']!='passed' for a in audits):raise ValueError('Three complete audited repeats required')
    if len({a['first200_ids_sha256'] for a in audits})!=1:raise ValueError('Different workloads')
    seconds=sum(a['cost']['continuous_wall_s'] for a in audits)
    energy=sum(a['cost']['estimated_system_energy_j'] for a in audits)
    board=sum(a['cost']['gpu_energy_j'] for a in audits)
    return dict(measured_calls=600,unique_test_videos=200,frames_per_video=4,
        seconds_total=seconds,sample_ms=seconds*1000/600,equivalent_frame_ms=seconds*1000/2400,
        sample_j=energy/600,equivalent_frame_j=energy/2400,time_weighted_board_w=board/seconds,
        common_input_tokens_total=sum(r['effective_input_tokens200'] for r in reports),
        measured200_srcc=[r['performance200']['srcc'] for r in reports],
        historical_full558_reference=reports[0]['original_full558_reference'],
        report_sha256=[a['report_sha256'] for a in audits],
        repeat_video_ms=[a['cost']['ms_per_video'] for a in audits],
        repeat_video_j=[a['cost']['estimated_joules_per_video'] for a in audits],
        full558_reference_not_new200_accuracy=True,no_fastest_repeat_selection=True)


def build(manifest):
    rows=[]
    for model in ('qwen','glm','clip','yolo'):
        for cap in (100,50):
            completed=[];pending=[];reports=[]
            for repeat,folder in enumerate(paths(model,cap),1):
                relative=str(folder.relative_to(REPO))
                if not (folder/'report.json').is_file() or not (folder/'process_audit.json').is_file():
                    pending.append(relative);continue
                report=entry.read(folder/'report.json')
                if (report['model'],report['cap'],report['repeat'])!=(model,cap,repeat):raise ValueError('Mapped point identity differs')
                audited=entry.audit_point(folder,manifest)
                bound=False
                for event in entry.read(folder/'process_audit.json'):
                    clients=set(event['clients'])
                    if clients-{report['pid']}:raise ValueError('Foreign MPS client in raw evidence')
                    if clients:bound=True
                    elif bound:raise ValueError('Own MPS client disappeared')
                if not bound:raise ValueError('Missing own MPS client')
                completed.append(dict(path=relative,**audited));reports.append(report)
            row=dict(model=model,cap=cap,status='complete' if len(completed)==3 else 'pending',
                qualified_completed=completed,pending_paths=pending,aggregate=None)
            if len(completed)==3:row['aggregate']=aggregate(reports,completed)
            rows.append(row)
    anchor=next(r for r in rows if r['model']=='qwen' and r['cap']==100)
    if anchor['status']!='complete':raise ValueError('Shared Qwen100 anchor must remain qualified')
    return dict(status='PARTIAL_NOT_FINAL_PAPER_TABLE',protocol_id=entry.PROTOCOL,
        shared_qwen_temporal100=anchor,rows=rows,remaining80gb_models=['DeepSeek-VL2 full','Kimi-VL-A3B'],
        rental_status='pending explicit new80GB machine from human',
        all_costs_include_raw_input_and_all_internal_gaps=True,old_cost_override=False,
        source_table_for_nine_task_temporal='shared_qwen_temporal100 in this exact index')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous index')
    result=build(args.manifest)
    with args.output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps([dict(model=r['model'],cap=r['cap'],status=r['status'],qualified=len(r['qualified_completed'])) for r in result['rows']]))


if __name__=='__main__':main()
