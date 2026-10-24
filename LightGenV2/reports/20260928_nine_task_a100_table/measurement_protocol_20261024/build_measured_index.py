"""One provenance index for both tables; missing new measurements stay missing.

This is an intermediate, machine-readable index, not a completed paper table.
The same immutable Qwen100 anchor must serve both tables. No historical narrow
cost is a fallback for missing input-inclusive measurements or OP counts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
from audit_t07 import audit,audit_qwen,read,sha
from audit_salicon import audit as audit_salicon
from audit_lsp import audit as audit_lsp
from audit_lgvq_spatial_qwen import audit as audit_spatial_qwen
from audit_abo_text_qwen import audit as audit_abo_text_qwen
from audit_abo_text_ops import audit as audit_abo_text_ops
from audit_abo_image_qwen import audit as audit_abo_image_qwen


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lsp-data-root',type=Path,default=REPO/'data/lsp_pose/lsp_dataset')
    parser.add_argument('--abo-data-root',type=Path,default=REPO/'data/abo_easy100_dataset_20260906')
    parser.add_argument('--abo-text-op-config',type=Path,default=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_operations_20261024_01/audit_original/model_config.json')
    parser.add_argument('--abo-image-python',type=Path,
        help='Optional existing CPU runtime matching the measured JPEG decoder; original auditor unchanged')
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
    spatial_root=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_spatial_input_a100_20261024_01'
    spatial=[audit_spatial_qwen(spatial_root/f'qwen_repeat{i:02d}',manifest) for i in (1,2,3)]
    seconds_sp=sum(r['cost']['continuous_wall_s'] for r in spatial)
    energy_sp=sum(r['cost']['estimated_system_energy_j'] for r in spatial)
    spatial_tokens=sum(r['common_input_tokens200'] for r in spatial)
    rows['01b']['qwen']=dict(status='independently_audited_600_calls',calls=600,unique_videos=200,frames_per_video=4,
        sample_ms=seconds_sp*1000/600,equivalent_frame_ms=seconds_sp*1000/2400,
        sample_j=energy_sp/600,equivalent_frame_j=energy_sp/2400,
        time_weighted_board_w=sum(r['cost']['gpu_energy_j'] for r in spatial)/seconds_sp,
        common_input_tokens_total=spatial_tokens,common_tokens_per_s=spatial_tokens/seconds_sp,
        full558_reference=spatial[0]['full558_reference'],measured200_subset=[r['measured200'] for r in spatial],
        raw_references=spatial,scope='file to CPU spatial score, continuous window, all199 gaps, zero dedicated inference warmup')
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
    lsp_root=REPO/'LightGenV2/tasks/t02_keypoint_detection/runs/simulation/a100_input_inclusive_20261024_02'
    lsp=audit_lsp(lsp_root,args.lsp_data_root)
    rows['05'].update(ours=lsp['aggregate']['ours'],qwen=lsp['aggregate']['qwen'],raw_audit=lsp,
                      status='matched_three_round_timing_energy_independently_audited; OP pending')
    common=600*196
    our_s=lsp['aggregate']['ours']['sample_ms']*.6;qwen_s=lsp['aggregate']['qwen']['sample_ms']*.6
    rows['05']['common_tokens']=dict(total=common,ours_per_s=common/our_s,qwen_per_s=common/qwen_s,
                                    rate_ratio=qwen_s/our_s,time_ratio=qwen_s/our_s,
                                    basis='196 common native visual input patches/image; original person crop and pose decoder')
    text_root=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_input_inclusive_20261024_02'
    text=[audit_abo_text_qwen(text_root/f'qwen_repeat{i:02d}',args.abo_data_root) for i in (1,2,3)]
    text_seconds=sum(v['continuous_wall_s'] for v in text);text_energy=sum(v['system_energy_j'] for v in text)
    text_tokens=sum(v['common_input_tokens200'] for v in text)
    rows['04']['qwen']=dict(status='independently_audited_600_calls',calls=600,unique_titles=100,
        sample_ms=1000*text_seconds/600,equivalent_frame_ms=1000*text_seconds/600,
        sample_j=text_energy/600,equivalent_frame_j=text_energy/600,
        native_unit='one title query, displayed as query/image-equivalent; no physical input frame',
        time_weighted_board_w=sum(v['board_energy_j'] for v in text)/text_seconds,
        common_input_tokens_total=text_tokens,common_tokens_per_s=text_tokens/text_seconds,
        full100_reference_hit1=.82,measured_unique100_hit1=[v['measured_unique100_hit1'] for v in text],
        raw_references=text,scope='UTF8 title to CPU2400 full ranking, two complete traversals per trial, all199 gaps')
    op_root=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_operations_20261024_01'
    text_ops=audit_abo_text_ops(op_root,args.abo_text_op_config)
    rows['04']['qwen'].update(raw_run_root=text_root.relative_to(REPO).as_posix(),
        operation_replay_root=op_root.relative_to(REPO).as_posix(),MAC_audit=text_ops,
        TOP_MAC_convention_per_query=text_ops['TOP_MAC_convention_per_query'],
        TOPS_MAC_convention=3*text_ops['MAC_OP200']/1e12/text_seconds,
        TOPS_per_W_MAC_convention=3*text_ops['MAC_OP200']/1e12/text_energy)
    rows['04']['ops_status']='Qwen actual200-input dense-MAC subtotal validated, explicit non-MAC/unsupported; Ours pending; not full scalar OP'
    # Immutable enrollment: two qualified original rounds, and only the missing
    # third from the continuation. Never enroll the externally rejected old r3.
    image_root=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation'
    image_paths=[image_root/'a100_image_input_inclusive_20261024_02'/f'qwen_repeat{i:02d}' for i in (1,2)]
    image_paths.append(image_root/'a100_image_input_inclusive_20261024_03/qwen_repeat03')
    def image_audit(path):
        if args.abo_image_python is None:return audit_abo_image_qwen(path,args.abo_data_root)
        import yaml
        # The existing local SciPy/NumPy audit runtime and measured JPEG decoder
        # have different Python ABIs. Execute the unchanged image auditor in its
        # explicitly selected existing runtime instead of replacing packages,
        # faking loadmat, relaxing pixel checks or rerunning any model.
        code=('import sys,json;from pathlib import Path;'
              'sys.path.insert(0,sys.argv[1]);'
              'sys.path.append(sys.argv[4]);'
              'from audit_abo_image_qwen import audit;'
              'print(json.dumps(audit(Path(sys.argv[2]),Path(sys.argv[3]))))')
        return json.loads(subprocess.check_output([str(args.abo_image_python),'-c',code,str(HERE),
            str(path.resolve()),str(args.abo_data_root.resolve()),str(Path(yaml.__file__).resolve().parent.parent)],cwd=REPO,text=True))
    images=[dict(raw_run_path=p.relative_to(REPO).as_posix(),**image_audit(p)) for p in image_paths]
    if [r['repeat'] for r in images]!=[1,2,3] or [r['common_input_tokens200'] for r in images]!=[13728]*3:
        raise ValueError('Original image-query200x3 enrollment changed')
    image_seconds=sum(r['continuous_wall_s'] for r in images);image_energy=sum(r['system_energy_j'] for r in images)
    image_tokens=sum(r['common_input_tokens200'] for r in images)
    rows['02']['qwen']=dict(status='independently_audited_600_calls',calls=600,unique_images=200,
        sample_ms=image_seconds/.6,equivalent_frame_ms=image_seconds/.6,sample_j=image_energy/600,
        equivalent_frame_j=image_energy/600,time_weighted_board_w=sum(r['board_energy_j'] for r in images)/image_seconds,
        common_input_tokens_total=image_tokens,common_tokens_per_s=image_tokens/image_seconds,
        full2400_reference_hit1=.7358333333333333,measured_first200_hit1=[r['measured_first200_hit1'] for r in images],
        independent_image_audit_python=str(args.abo_image_python or sys.executable),
        raw_references=images,scope='native image file to CPU100 full ranking, continuous199 gaps, zero dedicated inference warmup')
    rows['07']['status']='input_identity_conflict_pending_human_version_choice; no new inference'
    rows['07']['identity_note']='Selected .8880 PT579b19 uses old openmoji_grid_v2; .8120 Qwen PT0be775 uses layered scene TEST. Same numbered IDs are different input/target grids. Do not silently replace selected model or compare different datasets.'
    value=dict(protocol_id=protocol['protocol_id'],status='PARTIAL_NOT_FINAL_PAPER_TABLE',
        shared_qwen_temporal100=anchor,nine_metric_rows=list(rows.values()),
        reference_for_both_tables='shared_qwen_temporal100 in this exact index; never select another date or trial',
        current_input_protocol_sha256=sha(HERE/'protocol.json'),
        index_builder_source_sha256=sha(Path(__file__)),
        index_builder_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        no_predefined_speedup_claim=True,derived_optical_vs_measured_electronic_boundaries_separate=True,
        raw_data_in_original_task_runs=True,power_no_idle_subtraction=True)
    with args.output.open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=value['status'],shared_qwen100={k:anchor[k] for k in ('calls','sample_ms','sample_j','time_weighted_board_w')},
                          complete_timing_energy_rows=['03','05','06'],qualified_model_task_items=10,
                          pending_rows=['01a ours','01b ours','02 ours','04 ours','07 both: version choice','08 both'],
                          operation_counts='04 Qwen dense-MAC subtotal validated; other matching-input replays pending')))


if __name__=='__main__':main()
