"""Read sealed new pipeline raw; never run a model or connect to any host."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from lgvq_gpu_pipeline_20261026 import overlap,steady_report
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-4),(a,b)
def same_steady(a,b):
    assert a['criteria']==b['criteria'] and a['first_qualified_prefix_videos']==b['first_qualified_prefix_videos']
    assert len(a['all_windows'])==len(b['all_windows'])
    for x,y in zip(a['all_windows'],b['all_windows']):
        assert x['first_group']==y['first_group'] and x['last_group']==y['last_group']
        near(x['mean_ms'],y['mean_ms']);near(x['cv'],y['cv'])

def review_sealed_root(root,expected_status,receipt_name=None):
    state=read(root/'suite_status.json');assert state['status']==expected_status and state['mps_released']
    receipt_path=root/(receipt_name or ('backup_receipt_complete01.json' if expected_status=='complete' else 'backup_receipt_stopped01.json'))
    receipt=read(receipt_path);files=[]
    for item in receipt['verified_files']:
        p=root/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        files.append(dict(path=str(p.resolve()),bytes=item['bytes'],sha256=item['sha256']))
    for p in (receipt_path,root/'external_controller.log'):
        assert p.is_file()
        files.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
    return files


def review_leaf(folder):
    status=read(folder/'status.json');identity=read(folder/'identity.json')
    assert status['status']=='complete'
    assert identity['video_calls']==2048 and identity['unique_videos']==512
    receipt_path=folder/'backup_receipt_complete01.json';receipt_base=folder
    if receipt_path.exists():receipt=read(receipt_path)
    else:
        receipt_path=folder.parent/'backup_receipt_stopped01.json';receipt_base=folder.parent;receipt=read(receipt_path)
        receipt['verified_files']=[f for f in receipt['verified_files'] if f['path'].startswith(folder.name+'/')]
        assert receipt['verified_files']
    files=[]
    for f in receipt['verified_files']:
        p=receipt_base/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
        files.append(dict(path=str(p.resolve()),bytes=f['bytes'],sha256=f['sha256']))
    files.append(dict(path=str(receipt_path.resolve()),bytes=receipt_path.stat().st_size,sha256=sha(receipt_path)))
    for p,digest in identity['sources'].items():
        # The running server HEAD was intentionally not advanced across active
        # peer training. Resolve each actual source against the published commit.
        blob=subprocess.check_output(['git','show',identity['published_measurement_commit']+':'+p],cwd=REPO)
        assert hashlib.sha256(blob).hexdigest()==digest,p
    guard=read(folder/'process_audit.json');assert guard
    pid=identity['pid'];nonexclusive=identity.get('nonexclusive_authorized',False)
    platform_gaps=[];foreign_observations=[]
    for index,g in enumerate(guard):
        assert set(g['clients'])<={pid}
        if nonexclusive:
            assert g['human_authorized_nonexclusive']
            others=[x for x in g['processes'] if int(x[0].strip()) not in set(g['servers'])|{pid}]
            foreign_observations.append(dict(unix=g['unix'],foreign_processes=others,
                global_GPU_utilization_percent=g['global_GPU_utilization_percent'],global_memory_used_mib=g['global_memory_used_mib']))
        else:assert all(int(x[0].strip()) in set(g['servers'])|{pid} for x in g['processes'])
        if 'platform' in g:
            assert g['platform']['power_limit_w']==250 and g['platform']['uuid']==identity['gpu_uuid']
        else:
            # The original guard annotates audit[-1] after a subprocess query.
            # A final main-thread check can race its monitor thread. Preserve
            # this missing record, do not invent its platform fields.
            before=[x['platform'] for x in guard[:index] if 'platform' in x]
            after=[x['platform'] for x in guard[index+1:] if 'platform' in x]
            assert before and after and before[-1]==after[0]
            platform_gaps.append(dict(record_index=index,unix=g['unix'],platform_fields_missing=True,
                note='Recorded neighboring observations agree; this row was not repaired or filled. Source guard performs actual platform validation on every check.'))
    inputs=read(folder/'input_files.json');assert len(inputs)==512 and len({x['sample_id'] for x in inputs})==512
    prep=read(folder/'external_preparation.json')
    system=identity['system'];out=[];strict=False
    if system=='lg':
        consistency=read(folder/'consistency.json');assert consistency['status']=='passed'
        assert consistency['serial_pipeline_all_outputs_equal'] and consistency['original_first_full_field_output_equal']
        first_array=None
        for mode in ('serial','pipeline'):
            d=read(folder/(mode+'_timing.json'));records=d['records'];assert len(records)==128
            begin=records[0]['front_start'];end=records[-1]['body_end'];elapsed=end-begin
            near(elapsed,d['continuous_GPU_timeline_ms']);near(elapsed/128,d['time16_ms'])
            front_intervals=[];body_intervals=[];stage_totals=defaultdict(float);before_first_optic=0.;after_first_optic=0.;extra_GPU=0.
            minimal=d.get('instrumentation')=='minimal_boundary_events_no_per_module_timing'
            for i,r in enumerate(records):
                assert r['field']==i and r['front_start']<=r['front_end']<=r['body_start']<=r['body_end']
                if i:
                    assert r['front_start']>=records[i-1]['front_end']-.001
                    assert r['body_start']>=records[i-1]['body_end']-.001
                front_intervals.append((r['front_start'],r['front_end']));body_intervals.append((r['body_start'],r['body_end']))
                if minimal:
                    assert r['optical_passes']==6 and r['body_start']<=r['first_optic']<=r['body_end']
                    before_first_optic+=r['first_optic']-r['body_start']
                    after_first_optic+=r['body_end']-r['first_optic']
                else:
                    assert sum(s['family']=='optical_simulation_FFT' for s in r['body_stages'])==6
                    boundary=next(j for j,s in enumerate(r['body_stages']) if s['stage']=='parallel_optics.width_to_field')
                    for j,s in enumerate(r['body_stages']):
                        assert s['field']==i and s['cuda_event_ms']>=0
                        stage_totals[s['stage']]+=s['cuda_event_ms']
                        if s['charged']:
                            if j<boundary:before_first_optic+=s['cuda_event_ms']
                            else:after_first_optic+=s['cuda_event_ms']
                        else:extra_GPU+=s['cuda_event_ms']
                    assert {s['stage'] for s in r['front_stages']}=={'frontend.conv_embedding','frontend.image_position','frontend.mean_pool','frontend.text_embedding'}
                    for s in r['front_stages']:stage_totals[s['stage']]+=s['cuda_event_ms']
            ov=sum(overlap(front_intervals[i+1:],a,b) for i,(a,b) in enumerate(body_intervals))
            near(ov,d['actual_later_front_overlapping_body_ms'])
            if mode=='serial':near(ov,0)
            first_front_total=sum(b-a for a,b in front_intervals)
            starts=[r['body_end']-begin for r in records]
            same_steady(steady_report(starts),d['steady'])
            for p in d['prefixes']:near(p['time16_ms'],starts[p['video_calls']//16-1]*16/p['video_calls'])
            with np.load(folder/(mode+'_outputs.npz'),allow_pickle=False) as f:array=f['prediction'].copy()
            assert array.shape==(128,16) and np.isfinite(array).all()
            assert all(np.array_equal(array[:32],array[k*32:(k+1)*32]) for k in (1,2,3))
            if first_array is None:first_array=array
            else:assert np.array_equal(first_array,array)
            gpu_front=sum(stage_totals[k] for k in stage_totals if k.startswith('frontend.'))/128
            P=gpu_front+before_first_optic/128;E=after_first_optic/128;O=6*(1000/1400+.3+.0307)
            out.append(dict(system='LG',mode=mode,repeat=identity['repeat'],cap=100,video_calls=2048,
                elapsed_ms=elapsed,time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,
                observed_later_front_body_overlap_ms=ov,front_envelope_ms=first_front_total,
                overlapped_fraction_of_front_envelopes=ov/first_front_total,
                first_qualified_steady_videos=d['steady']['first_qualified_prefix_videos'],prefixes=d['prefixes'],
                instrumentation=d.get('instrumentation','dense_module_events'),
                front_envelope16_ms=first_front_total/128,input_adapters_body_envelope16_ms=before_first_optic/128,
                first_optic_to_score_software_envelope16_ms=after_first_optic/128,
                stages_per16_ms=None if minimal else {k:v/128 for k,v in stage_totals.items()},
                theoretical_only=None if minimal else dict(P_GPU_before_first_optics_ms=P,E_after_first_optics_ms=E,O_fixed_ms=O,
                    serial_segment_projection16_ms=P+E+O,independent_resources_cycle_ms=max(P,E+O),
                    warning='Segment sum has incomplete scope coverage and mixed-stream envelopes. Not measured physical E2E or verified hardware overlap.'),
                original_path=str((folder/(mode+'_timing.json')).resolve())))
    else:
        d=read(folder/'qwen_timing.json');records=d['records'];assert len(records)==2048
        strict=d.get('execution')=='strict_single_video_CPU_then_H2D_then_GPU_then_CPU_return'
        if strict:
            assert d['no_prefetch'] and d['no_prepared_input_reuse'] and d['batch_videos']==1
            assert identity['CPU_preparation_one_video_at_a_time'] and identity['no_prefetch']
            for i,r in enumerate(records):
                assert r['CPU_start']<=r['CPU_end']<=r['H2D_end']<=r['host_start']<r['host_end']<=r['CPU_result_return']
                if i:assert r['CPU_start']>=records[i-1]['CPU_result_return']
        elapsed=records[-1]['end_ms']-records[0]['start_ms'];near(elapsed,d['continuous_GPU_timeline_ms'])
        assert abs(elapsed-d['continuous_host_ms'])<max(10,d['continuous_host_ms']*.001)
        assert d['all_inter_video_waits_included'] and d['terminal_CPU_copy_excluded']
        for i,r in enumerate(records):
            assert r['index']==i and r['source_index']==i%512 and r['cycle']==i//512
            assert r['start_ms']<=r['first_TF_ms']<r['end_ms']
            if i:assert r['start_ms']>=records[i-1]['end_ms']-.001
        work=sum(r['end_ms']-r['start_ms'] for r in records)
        gaps=sum(b['start_ms']-a['end_ms'] for a,b in zip(records,records[1:]))
        near(work+gaps,elapsed);near(elapsed/128,d['time16_ms'])
        with np.load(folder/'qwen_outputs.npz',allow_pickle=False) as f:array=f['prediction'].copy()
        assert array.shape==(4,512) and np.isfinite(array).all() and all(np.array_equal(array[0],a) for a in array[1:])
        groupends=[records[i]['end_ms'] for i in range(15,2048,16)]
        if not strict:same_steady(steady_report(groupends),d['steady'])
        pre=sum(r['first_TF_ms']-r['start_ms'] for r in records)/128
        post=sum(r['end_ms']-r['first_TF_ms'] for r in records)/128
        out.append(dict(system='Qwen',mode='native_serial',repeat=identity['repeat'],cap=50,video_calls=2048,
            elapsed_ms=elapsed,time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,
            model_front16_ms=pre,first_TF_to_score16_ms=post,inter_video_gaps16_ms=gaps/128,
            strict_single_video_CPU_GPU_serial=strict,nonexclusive_authorized=nonexclusive,
            not_uncontended_baseline_performance=nonexclusive,
            GPU_model_window_sum16_ms=work/128,first_TF_window_sum16_ms=post,
            first200_first_TF_window16_ms=sum(r['end_ms']-r['first_TF_ms'] for r in records[:200])/200*16,
            file_to_last_CPU_result16_ms=d.get('file_input_to_last_CPU_result_ms',0)/128 if strict else None,
            first_qualified_steady_videos=None if strict else d['steady']['first_qualified_prefix_videos'],
            steady_state_not_applicable_to_Qwen_strict_serial=strict,prefixes=d['prefixes'],
            original_path=str((folder/'qwen_timing.json').resolve())))
    tokenize=prep.get('tokenizer_calls',prep.get('tokenizer',[]))
    extra=dict(system=system,repeat=identity['repeat'],recorded_CPU_tokenizer_ms=sum((r['end']-r['start'])*1000 for r in tokenize),
        tokenizer_call_count=len(tokenize),prepared_unique_inputs=512,actual_CPU_prepared_video_calls=2048 if strict else 512,
        preparation_reused_as_raw_input_across4cycles=not strict,
        guard_platform_annotation_gaps=platform_gaps,
        human_authorized_nonexclusive=nonexclusive,foreign_observations=foreign_observations,
        wall_note=('Qwen CPU preparation and H2D occur per video, interleaved with its inference. All but first preparation are INSIDE the continuous timeline.' if strict else
            'External CPU preparation and H2D were separately observed once per512 raw inputs, not part of LG GPU-input-resident primary timeline.'))
    if strict and identity.get('tokenizer_is_CPU_separate_not_added_to_GPU_timeline'):
        extra['scope_metadata_warning']=dict(legacy_inherited_flag='tokenizer_is_CPU_separate_not_added_to_GPU_timeline=True',
            incorrect_for_new_strict_serial_scope=True,original_identity_not_rewritten=True,
            actual_scope='Qwen tokenizer is part of each serial CPU preparation and included in all intermediate continuous gaps; explicit strict execution/profile and recorded boundaries govern.')
    if system=='lg':
        tasks=[t for group in prep['rows'] for t in group['CPU_preparation']]
        extra.update(CPU_worker_work_sum_ms=dict(decode=sum((t['decode_end']-t['start'])*1000 for t in tasks),
            image_processor=sum((t['processor_end']-t['decode_end'])*1000 for t in tasks),
            quality14=sum((t['end']-t['processor_end'])*1000 for t in tasks)),
            CPU_preparation_span_ms=(max(t['end'] for t in tasks)-min(t['start'] for t in tasks))*1000,
            packing_transfer_tokenizer_window_sum_ms=sum((t['end']-t['packing_transfer_and_tokenizer_start'])*1000 for t in prep['rows']),
            work_sum_is_NOT_wall_time=True,external_stages_overlap_and_must_not_be_added=True)
    else:extra.update(CPU_prepare_total_ms=sum((t['CPU_end']-t['CPU_start'])*1000 for t in prep['rows']),
            H2D_total_ms=sum((t['H2D_end']-t['CPU_end'])*1000 for t in prep['rows']),tokenizer_already_in_CPU_prepare=True)
    return out,inputs,files,extra


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--lg-root',type=Path);p.add_argument('--diagnostic-root',type=Path);p.add_argument('--retained-qwen-root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.retained_qwen_root:
        old=read(a.retained_qwen_root/'suite_status.json');new=read(a.root/'suite_status.json')
        assert old['status']=='stopped' and old['completed']==['qwen_repeat01']
        assert new['status']=='complete' and new['plan']==[2,3] and new['completed']==['qwen_repeat02','qwen_repeat03']
        kept=new['retained_first_qualified_point']
        retained_manifest=a.root.with_suffix('.retained01.json')
        assert retained_manifest.is_file() and sha(retained_manifest)==kept['manifest_sha256']
        assert read(retained_manifest)['verified_files']==kept['verified_files']
        assert kept['timing_sha256']==sha(a.retained_qwen_root/'qwen_repeat01/qwen_timing.json')
        for item in kept['verified_files']:
            member=a.retained_qwen_root/'qwen_repeat01'/item['path']
            assert member.stat().st_size==item['bytes'] and sha(member)==item['sha256']
    points=[];sources=[];inputs_ref=None;extras=[];outputs_ref={}
    for system in ('lg','qwen'):
        for rep in (1,2,3):
            root=(a.lg_root or a.root) if system=='lg' else (a.retained_qwen_root if rep==1 and a.retained_qwen_root else a.root)
            pts,inputs,files,extra=review_leaf(root/f'{system}_repeat{rep:02d}');points+=pts;sources+=files;extras.append(extra)
            prediction_file=root/f'{system}_repeat{rep:02d}'/('pipeline_outputs.npz' if system=='lg' else 'qwen_outputs.npz')
            with np.load(prediction_file,allow_pickle=False) as saved:prediction=saved['prediction'].copy()
            if system not in outputs_ref:outputs_ref[system]=prediction
            else:assert np.array_equal(outputs_ref[system],prediction),'Fresh-repeat full outputs differ: '+system
            key=[(r['sample_id'],r['sha256']) for r in inputs]
            if inputs_ref is None:inputs_ref=key
            else:assert key==inputs_ref,'Models/repeats did not use identical512 original video bytes'
    aggregate=[]
    for system,mode in [('LG','serial'),('LG','pipeline'),('Qwen','native_serial')]:
        group=[r for r in points if r['system']==system and r['mode']==mode];assert len(group)==3
        total=sum(r['elapsed_ms'] for r in group);mean=total/384
        aggregate.append(dict(system=system,mode=mode,total_video_calls=6144,total_elapsed_ms=total,time16_ms=mean,
            equivalent_frame_ms=mean/64,repeat_time16_ms=[r['time16_ms'] for r in group],
            steady_videos=[r['first_qualified_steady_videos'] for r in group],
            actual_overlap_per_run_ms=[r.get('observed_later_front_body_overlap_ms',0) for r in group]))
    q=aggregate[-1]['time16_ms']
    for r in aggregate:r['Qwen50_over_system_ratio']=q/r['time16_ms']
    strict_Qwen=[r for r in points if r['system']=='Qwen']
    assert all(r['strict_single_video_CPU_GPU_serial'] for r in strict_Qwen),'Resident-prepared Qwen is not the authorized normal comparison'
    q_gpu=sum(r['GPU_model_window_sum16_ms'] for r in strict_Qwen)/3
    nonexclusive=any(r.get('nonexclusive_authorized',False) for r in strict_Qwen)
    for r in aggregate:r['Qwen50_GPU_model_windows_over_system_ratio']=q_gpu/r['time16_ms']
    diagnostics=[]
    if a.diagnostic_root:
        for rep in (1,2,3):
            pts,inputs,files,extra=review_leaf(a.diagnostic_root/f'lg_repeat{rep:02d}');diagnostics+=pts;sources+=files
            assert [(r['sample_id'],r['sha256']) for r in inputs]==inputs_ref
    for root in {a.root,a.lg_root or a.root}:sources+=review_sealed_root(root,'complete')
    if a.retained_qwen_root:sources+=review_sealed_root(a.retained_qwen_root,'stopped','backup_receipt_MPS_control_stop01.json')
    if a.retained_qwen_root:sources.append(dict(path=str(retained_manifest.resolve()),bytes=retained_manifest.stat().st_size,sha256=sha(retained_manifest)))
    if a.diagnostic_root:sources+=review_sealed_root(a.diagnostic_root,'stopped')
    sources=list({f['path']:f for f in sources}.values())
    result=dict(status='passed',matched_video_calls_per_repeat=2048,unique_original_videos=512,cycles=4,
        accuracy_not_re_evaluated=True,physical_optical_timing_not_measured=True,
        boundary='LG optimized raw-GPU-input pipeline; Qwen strict per-video CPU-preparation/H2D/forward/CPU-return. Continuous first GPU frontend through last GPU score, ALL intermediate waits retained.',
        Qwen_GPU_model_window_sum16_ms=q_gpu,
        LG_CPU_file_preparation_not_in_primary=True,Qwen_inter_video_CPU_preparation_in_primary=True,
        comparison_is_requested_execution_strategies_not_symmetric_file_input_E2E=True,
        Qwen_nonexclusive_authorized=nonexclusive,
        ratios_eligible_for_uncontended_hardware_speedup_claim=not nonexclusive,
        interference_warning=('Qwen shared the physical GPU with uncontrolled other applications. LG retained data were uncontended; resulting ratios are exploratory load-dependent references, not an uncontended hardware speedup.' if nonexclusive else None),
        all_points=points,aggregates=aggregate,extra_preparation=extras,LG_dense_diagnostics=diagnostics,
        actual_input_hash_sequence_verified=True,measured_overlap_is_software_body_not_physical_6cycle_coverage=True,
        all_fresh_repeat_full_predictions_bitwise_equal_within_each_model=True,
        metadata_warnings=[p['scope_metadata_warning'] for p in extras if 'scope_metadata_warning' in p],
        source_files=sources)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    import csv
    with (a.output/'raw_file_index.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(sources)
    print(json.dumps(dict(status='passed',aggregates=aggregate,output=str(a.output)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
