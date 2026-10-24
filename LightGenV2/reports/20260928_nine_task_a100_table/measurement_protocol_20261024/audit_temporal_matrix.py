"""Fixed canonical run map for both temporal tables; no 'latest'/best selection.

Only independently replayed complete triples get aggregate values. Pending
points stay missing. This CPU index is not the final paper Excel or OP count.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
RENTAL_GPU='GPU-6d1916d1-fbd3-524b-3d98-7fb38512c388'
RENTAL_RUN='rental_80gb_20261025_01'
ANCHOR_RUN='rental_qwen_anchor_20261025_02'


def paths(model,cap):
    root=entry.CANONICAL
    if model in ('deepseek','kimi'):
        return [root/RENTAL_RUN/f'{model}_cap{cap}_repeat{i:02d}' for i in (1,2,3)]
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


def platform(model):
    return dict(gpu_uuid=RENTAL_GPU,gpu_name='NVIDIA A100-SXM4-80GB',power_limit_w=400,
        group='new_authorized_rental80GB') if model in ('deepseek','kimi','rental_qwen_anchor') else dict(
        gpu_uuid='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55',gpu_name='NVIDIA A100-PCIE-40GB',
        power_limit_w=250,group='original_laboratory40GB')


def audit_rental_anchor(manifest):
    root=entry.CANONICAL/ANCHOR_RUN;completed=[];reports=[];pending=[]
    for repeat in (1,2,3):
        folder=root/f'qwen_cap100_repeat{repeat:02d}'
        if not (folder/'rental_anchor_role.json').exists():
            pending.append(folder.relative_to(REPO).as_posix());continue
        role=entry.read(folder/'rental_anchor_role.json');report=entry.read(folder/'report.json')
        if (role['role']!='supplementary_same_80GB_GPU_Qwen_anchor_only'
                or not role['not_the_shared_40GB_table_anchor'] or role['report_sha256']!=entry.sha(folder/'report.json')
                or report['gpu_uuid']!=RENTAL_GPU or report['power_limit_w']!=400
                or (report['model'],report['cap'],report['repeat'])!=('qwen',100,repeat)):
            raise ValueError('Same-new-GPU Qwen supplementary anchor role/identity mismatch')
        completed.append(dict(path=folder.relative_to(REPO).as_posix(),**entry.audit_point(folder,manifest)))
        reports.append(report)
    return dict(status='complete' if len(completed)==3 else 'pending',
        role='supplementary_same80GB_Qwen_control_not_replacement_of_shared40GB_anchor',
        platform=platform('rental_qwen_anchor'),qualified_completed=completed,pending_paths=pending,
        aggregate=aggregate(reports,completed) if len(completed)==3 else None)


def build(manifest):
    rows=[]
    for model in ('qwen','glm','clip','yolo','deepseek','kimi'):
        for cap in (100,50):
            completed=[];pending=[];reports=[]
            for repeat,folder in enumerate(paths(model,cap),1):
                relative=str(folder.relative_to(REPO))
                if not (folder/'report.json').is_file() or not (folder/'process_audit.json').is_file():
                    pending.append(relative);continue
                report=entry.read(folder/'report.json')
                if (report['model'],report['cap'],report['repeat'])!=(model,cap,repeat):raise ValueError('Mapped point identity differs')
                expected=platform(model)
                if (report['gpu_uuid']!=expected['gpu_uuid'] or report['gpu_name']!=expected['gpu_name']
                        or report['power_limit_w']!=expected['power_limit_w']):
                    raise ValueError('Do not mix different physical platforms in a triple')
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
                qualified_completed=completed,pending_paths=pending,aggregate=None,platform=platform(model))
            if len(completed)==3:row['aggregate']=aggregate(reports,completed)
            rows.append(row)
    anchor=next(r for r in rows if r['model']=='qwen' and r['cap']==100)
    from lgvq_qwen_shape_receipt import attach as attach_video_qwen_MAC
    from rental_shape_receipt import attach as attach_original_rental_MAC
    for row in rows:
        if row['model']=='qwen' and row['aggregate'] is not None:
            attach_video_qwen_MAC(row['aggregate'],f"laboratory40GB_cap{row['cap']}")
        elif row['model'] in ('deepseek','kimi') and row['aggregate'] is not None:
            attach_original_rental_MAC(row['aggregate'],row['model'],row['cap'])
    if anchor['status']!='complete':raise ValueError('Shared Qwen100 anchor must remain qualified')
    common=anchor['aggregate']['common_input_tokens_total']
    if common!=89200*3:raise ValueError('Original shared Qwen first200 token work differs')
    for row in rows:
        if row['aggregate'] is not None:
            # Preserve each model's original native-token field in the old
            # aggregate. Only this explicitly named field is common workload.
            row['common_Qwen_input_work']=dict(valid_nonpadding_input_tokens600=common,
                tokens_per_s=common/row['aggregate']['seconds_total'],
                source='same first200 IDs as fixed shared40GB Qwen100; same numerator for all rows',
                not_generation_throughput=True)
    remaining=[name for name in ('deepseek','kimi') if any(r['status']!='complete' for r in rows if r['model']==name)]
    supplementary=audit_rental_anchor(manifest)
    if supplementary['aggregate'] is not None:
        attach_video_qwen_MAC(supplementary['aggregate'],'supplementary80GB_cap100')
    return dict(status='PARTIAL_NOT_FINAL_PAPER_TABLE',protocol_id=entry.PROTOCOL,
        shared_qwen_temporal100=anchor,rows=rows,remaining80gb_models=remaining,
        rental_status='new authorized80GB costs complete' if not remaining else 'new authorized80GB missing trials remain',
        supplementary_rental_qwen_anchor=supplementary,
        platform_comparison='40GB PCIe/250W and80GB SXM/400W separately identified; not a common-hardware causal model ranking',
        all_costs_include_raw_input_and_all_internal_gaps=True,old_cost_override=False,
        formal_accuracy='confirmed original full558 only; measured200 predictions are diagnostics',
        index_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
