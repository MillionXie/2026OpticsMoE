"""One provenance index for both tables; missing new measurements stay missing.

This is an intermediate, machine-readable index, not a completed paper table.
The same immutable Qwen100 anchor must serve both tables. No historical narrow
cost is a fallback for missing input-inclusive measurements or OP counts.
"""
from __future__ import annotations
import argparse
import csv
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
from audit_abo_title_student import audit as audit_abo_title_student
from audit_openmoji_e45 import audit as audit_openmoji_e45
from audit_openmoji_qwen import audit as audit_openmoji_qwen
from protocol_identity import effective_protocol


def title_student_paths(root):
    # The two failed attempts remain raw evidence, never candidates for selection.
    return [root/'a100_title_student_input_inclusive_20261024_02/ours_repeat01',
            root/'a100_title_student_input_inclusive_20261025_04/ours_repeat02',
            root/'a100_title_student_input_inclusive_20261025_04/ours_repeat03']


def title_student_result(root,data_root):
    results=[dict(raw_run_path=p.relative_to(REPO).as_posix(),**audit_abo_title_student(p,data_root))
             for p in title_student_paths(root)]
    if ([r['repeat'] for r in results]!=[1,2,3] or [r['calls'] for r in results]!=[200]*3
            or [r['common_input_tokens200'] for r in results]!=[9576]*3):
        raise ValueError('Canonical original title student200x3 enrollment changed')
    seconds=sum(r['projected_seconds200'] for r in results)
    energy=sum(r['phase_estimated_j200'] for r in results)
    tokens=sum(r['common_input_tokens200'] for r in results)
    return dict(status='independently_audited_600_calls',calls=600,unique_titles=100,
        sample_ms=seconds/.6,equivalent_frame_ms=seconds/.6,sample_j=energy/600,equivalent_frame_j=energy/600,
        common_input_tokens_total=tokens,common_tokens_per_s=tokens/seconds,
        full100_historical_reference_hit1=.86,measured_unique100_hit1=[r['measured_unique100_hit1'] for r in results],
        optical_ms_total=sum(r['optical_ms_total'] for r in results),
        charged_electronic_ms_total=sum(r['charged_electronic_ms_total'] for r in results),
        actual_software_seconds_total=sum(r['software_wall_s'] for r in results),
        native_unit='one UTF8 title query; image-equivalent display, no physical query frame',
        optical_passes_per_query=3,online_visual_or_bridge_calls=0,
        bridge_zero_reason=results[0]['bridge_zero_reason'],not_a_device_measured_e2e=True,
        raw_references=results,
        scope='real UTF8/tokenizer/H2D, selected original NN synchronized wall plus3 parameter-derived optical passes, CPU full2400 ranking and all199 true gaps; separate offline gallery')


def openmoji_e45_paths(root):
    # Only the complete three fresh processes, never a frontend-only contract.
    return [root/'a100_e45_input_inclusive_20261025_01'/f'ours_repeat{i:02d}' for i in (1,2,3)]


def openmoji_e45_result(root):
    results=[dict(raw_run_path=p.relative_to(REPO).as_posix(),**audit_openmoji_e45(p))
             for p in openmoji_e45_paths(root)]
    if ([r['repeat'] for r in results]!=[1,2,3] or [r['calls'] for r in results]!=[200]*3
            or any(r['common_Qwen_input_tokens200'] is not None for r in results)):
        raise ValueError('Canonical new-layout epoch45 workload or pending common token basis changed')
    seconds=sum(r['projected_seconds200'] for r in results)
    energy=sum(r['phase_estimated_j200'] for r in results)
    return dict(status='independently_audited_600_calls',calls=600,unique_images=200,
        sample_ms=seconds/.6,equivalent_frame_ms=seconds/.6,sample_j=energy/600,equivalent_frame_j=energy/600,
        full1000_historical_simulation_changed_cell_accuracy=.8765,
        full1000_historical_laboratory_changed_cell_accuracy=.8655,
        measured_first200_metrics=[r['measured_first200_metrics'] for r in results],
        common_input_tokens_total=None,common_tokens_per_s=None,
        common_tokens_status='pending actual same-input Qwen formal calls; not Ours plain internal tokens',
        optical_ms_total=sum(r['optical_ms_total'] for r in results),
        charged_electronic_ms_total=sum(r['charged_electronic_ms_total'] for r in results),
        diagnostic_actual_software_seconds_total=sum(r['diagnostic_software_wall_s'] for r in results),
        diagnostic_actual_software_board_j_total=sum(r['diagnostic_software_board_j200'] for r in results),
        native_unit='one original PNG plus UTF8 instruction to CPU semantic grid',
        optical_passes_per_query=6,not_a_device_measured_e2e=True,raw_references=results,
        scope='real PNG/plain tokenizer/BF16 embedding, original selected NN synchronized wall plus6 parameter-derived optical passes, original CPU grid result and all199 true gaps; original residual/FFT/layout retained separately')


def openmoji_qwen_paths(root):
    return [root/'a100_qwen_input_inclusive_20261025_01'/f'qwen_repeat{i:02d}' for i in (1,2,3)]


def openmoji_pair(root,ours):
    paths=openmoji_qwen_paths(root)
    results=[dict(raw_run_path=p.relative_to(REPO).as_posix(),**audit_openmoji_qwen(p)) for p in paths]
    if ([v['repeat'] for v in results]!=[1,2,3] or [v['calls'] for v in results]!=[200]*3
            or [v['common_Qwen_input_tokens200'] for v in results]!=[12807]*3):
        raise ValueError('Canonical native OpenMoji200x3 enrollment differs')
    # Derive common work in the new index, never edit the sealed Ours raw that
    # correctly recorded "pending" before the native trials existed.
    for own_path,qwen_path,result in zip(openmoji_e45_paths(root),paths,results):
        def calls(path):
            with (path/'calls.csv').open(encoding='utf-8-sig',newline='') as stream:
                return list(csv.DictReader(stream))
        own,native=calls(own_path),calls(qwen_path)
        if len(own)!=200 or len(native)!=200:raise ValueError('Missing matched cost queries')
        for a,b in zip(own,native):
            if any(a[k]!=b[k] for k in ('index','sample_id','task','instruction','instruction_utf8_sha256')):
                raise ValueError('Different OpenMoji input identities; internal plain tokens are not common native work')
        if sum(int(c['valid_input_tokens']) for c in native)!=result['common_Qwen_input_tokens200']:
            raise ValueError('Common work must be actual native valid input IDs/masks')
    seconds=sum(v['continuous_wall_s'] for v in results);energy=sum(v['system_energy_j'] for v in results)
    tokens=sum(v['common_Qwen_input_tokens200'] for v in results);own_seconds=ours['sample_ms']*.6
    ours.update(common_input_tokens_total=tokens,common_tokens_per_s=tokens/own_seconds,
        common_tokens_status='derived from independently audited same-input native Qwen trials; sealed Ours raw unchanged')
    native=dict(status='independently_audited_600_calls',calls=600,unique_images=200,
        sample_ms=seconds/.6,equivalent_frame_ms=seconds/.6,sample_j=energy/600,equivalent_frame_j=energy/600,
        time_weighted_board_w=sum(v['board_energy_j'] for v in results)/seconds,
        common_input_tokens_total=tokens,common_tokens_per_s=tokens/seconds,
        formal_full1000_reference_metrics=results[0]['formal_full1000_reference_metrics'],
        measured_first200_diagnostics_only=[v['measured_first200_diagnostics_only'] for v in results],
        raw_references=results,scope='original PNG/native chat full Vision+Language+CPU features+head to CPU grid, all199 gaps, zero dedicated inference warmup')
    common=dict(total=tokens,ours_per_s=tokens/own_seconds,qwen_per_s=tokens/seconds,
        rate_ratio=seconds/own_seconds,time_ratio=seconds/own_seconds,
        basis='Same original PNG and UTF8 instruction per first200; actual Qwen valid native input IDs/masks, not Ours plain internal tokens')
    return native,common


def attach_openmoji_existing_shape_MAC(row):
    """Consume the new zero-forward CPU derivations and their independent copy.

    Never label the Ours NN subtotal as total optical dense-equivalent work.
    Both physical optics and non-MAC scalar work remain separately explicit.
    """
    files={'qwen':'openmoji_qwen_network_shape_MAC','ours':'openmoji_e45_saved_shape_MAC'}
    for model,stem in files.items():
        path=HERE/(stem+'_01.json');server=HERE/(stem+'_server_01.json');value=read(path)
        if value!=read(server) or value['calls']!=600 or value['complete_scalar_OP_claim']:
            raise ValueError('Independent same-scope NN MAC derivations differ')
        elapsed=row[model]['sample_ms']*.6;energy=row[model]['sample_j']*600
        if abs(sum(r['seconds200'] for r in value['cost_references'])-elapsed)>1e-8:
            raise ValueError('NN OP numerator not matched to original time denominator')
        if abs(sum(r['joules200'] for r in value['cost_references'])-energy)>1e-7:
            raise ValueError('NN OP numerator not matched to original energy denominator')
        if model=='qwen':
            op=value['MAC_OP600']
            if op!=2*sum(r['MAC200'] for r in value['repeat_results']) or value['GPU_forwards']!=0:
                raise ValueError('Original native shape-derived numerator differs')
        else:
            op=value['actual_NN_MAC_OP600']
            if (op!=2*sum(r['actual_executed_NN_MAC200'] for r in value['repeat_results'])
                    or not value['no_GPU_or_neural_forwards'] or value['physical_optical_dense_equivalent_OP'] is not None):
                raise ValueError('NN-only subtotal scope/unknown physical OP changed')
        row[model]['derived_NN_dense_MAC_subtotal']=dict(OP600=op,TOP_per_input=op/600/1e12,
            TOPS=op/1e12/elapsed,TOPS_per_W=op/1e12/energy,
            source='original network + sealed actual input/module shapes; zero new model forward',
            local_receipt=path.relative_to(REPO).as_posix(),local_receipt_sha256=sha(path),
            server_receipt=server.relative_to(REPO).as_posix(),server_receipt_sha256=sha(server),
            not_complete_scalar_OP=True,physical_optical_dense_equivalent_not_in_NN_subtotal=True,
            raw_derivation=value)
    row['ops_status']='Both NN dense-MAC subtotals derived from original graph/existing all200 shapes, independent two-end CPU agreement, zero new GPU; optical dense-equivalent and non-MAC separately explicit'


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
    protocol=effective_protocol()
    rows={item['id']:dict(id=item['id'],task=item['task'],metric=item['metric'],
                         full_test_reference=dict(ours=item['ours_target'],qwen=item['qwen_target']),
                         ours=None,qwen=None,ops_status='pending derivation from original network and existing executed shapes; acquire only missing shapes, no mandatory GPU replay',
                         status='not_yet_complete_under_new_protocol') for item in protocol['rows']}
    for item in protocol['rows']:
        if 'ours_lab_target' in item:
            rows[item['id']]['full_test_reference']['ours_lab']=item['ours_lab_target']
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
    student_root=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation'
    rows['04']['ours']=title_student_result(student_root,args.abo_data_root)
    if rows['04']['ours']['common_input_tokens_total']!=text_tokens:
        raise ValueError('Matched title systems must use the same actual input token work')
    our_seconds=rows['04']['ours']['sample_ms']*.6
    rows['04']['common_tokens']=dict(total=text_tokens,ours_per_s=text_tokens/our_seconds,
        qwen_per_s=text_tokens/text_seconds,rate_ratio=text_seconds/our_seconds,time_ratio=text_seconds/our_seconds,
        basis='Same original Qwen input IDs/masks per actual title; not generated tokens or separate internal work')
    rows['04']['status']='matched_three_round_timing_energy_independently_audited; Qwen dense-MAC subtotal validated, Ours OP pending'
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
    openmoji_root=REPO/'LightGenV2/tasks/t04_semantic_interaction/runs/simulation'
    rows['07']['ours']=openmoji_e45_result(openmoji_root)
    rows['07']['qwen'],rows['07']['common_tokens']=openmoji_pair(openmoji_root,rows['07']['ours'])
    attach_openmoji_existing_shape_MAC(rows['07'])
    rows['07']['status']='matched_three_round_timing_energy_independently_audited; original-network shape OP derivation pending'
    rows['07']['identity_note']='Human selected the new-layout laboratory .8655 version: epoch45/PT03cb861c, same-weight simulation .8765, Qwen epoch30/PT0be775 .8120. New first200 Ours metrics are separate from both full1000 references. Old .8880/PT579b19 and original protocol remain historical; no new laboratory capture or paired per-case CCD audit is implied.'
    rows['07']['effective_identity']=next(item for item in protocol['rows'] if item['id']=='07')
    value=dict(protocol_id=protocol['protocol_id'],status='PARTIAL_NOT_FINAL_PAPER_TABLE',
        shared_qwen_temporal100=anchor,nine_metric_rows=list(rows.values()),
        reference_for_both_tables='shared_qwen_temporal100 in this exact index; never select another date or trial',
        current_input_protocol_sha256=sha(HERE/'protocol.json'),
        approved_identity_amendments=protocol['identity_amendments'],
        reporting_policy=dict(formal_accuracy='confirmed original full TEST results only',
            cost_workload='first200 TEST calls x3 fresh processes; subset prediction metrics are diagnostics only',
            operation_counts='derive from original network and existing actual shape records; only supplement missing shapes',
            TOPS_formula='OP / (1e12 * matched elapsed seconds)',
            TOPS_per_W_formula='OP / (1e12 * matched energy joules)',
            no_retraining_or_rerun_of_qualified_trials=True),
        index_builder_source_sha256=sha(Path(__file__)),
        index_builder_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        no_predefined_speedup_claim=True,derived_optical_vs_measured_electronic_boundaries_separate=True,
        raw_data_in_original_task_runs=True,power_no_idle_subtraction=True)
    with args.output.open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=value['status'],shared_qwen100={k:anchor[k] for k in ('calls','sample_ms','sample_j','time_weighted_board_w')},
                          complete_timing_energy_rows=['03','04','05','06','07'],
                          qualified_model_task_items=sum(r[m] is not None for r in rows.values() for m in ('ours','qwen')),
                          pending_rows=['01a ours','01b ours','02 ours','08 both'],
                          operation_counts='04 Qwen and07 both NN dense-MAC subtotals verified; optical dense-equivalent separate; other original-network/existing-shape derivations pending, no compulsory GPU replay')))


if __name__=='__main__':main()
