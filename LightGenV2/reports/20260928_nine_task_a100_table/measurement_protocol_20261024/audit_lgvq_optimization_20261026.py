"""Offline review of sealed paired LG experiments. Never imports a GPU model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-3),(a,b)
def record(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))

def sealed(root,status,receipt):
    state=read(root/'suite_status.json');assert state['status']==status and state['mps_released']
    receipt_path=root/receipt;files=[]
    for row in read(receipt_path)['verified_files']:
        p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
        files.append(record(p))
    files.extend(record(p) for p in (receipt_path,root/'external_controller.log'))
    return files

def guard(folder,identity):
    values=read(folder/'process_audit.json');assert values
    missing=[]
    for i,row in enumerate(values):
        assert row['human_authorized_nonexclusive']
        assert set(row['clients'])<={identity['pid']} and not row['foreign_mps_clients']
        if 'platform' not in row:
            before=[r['platform'] for r in values[:i] if 'platform' in r]
            after=[r['platform'] for r in values[i+1:] if 'platform' in r]
            assert before and after and before[-1]==after[0]
            missing.append(i);continue
        platform=row['platform']
        assert platform['uuid']==identity['gpu_uuid'] and platform['power_limit_w']==250 and platform['requested_cap']==100
    return dict(nonexclusive=True,foreign_observation_count=sum(bool(r.get('foreign_gpu_processes')) for r in values),
                records=len(values),unfilled_platform_annotation_rows=missing)

def original_sources(folder,identity,commit):
    recorded=identity.get('published_measurement_commit')
    # A missing launch environment tag is never silently filled into raw.
    selected=recorded or commit
    assert selected
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',selected+':'+name],cwd=REPO)
        assert hashlib.sha256(blob).hexdigest()==digest,name
    return dict(recorded_published_commit=recorded,verified_published_commit=selected,
                missing_recorded_commit_not_rewritten=recorded is None,source_count=len(identity['sources']))

def predictions(folder,name):
    with np.load(folder/name,allow_pickle=False) as f:result=f['prediction'].copy()
    assert result.shape==(128,16) and np.isfinite(result).all()
    assert all(np.array_equal(result[:32],result[k*32:(k+1)*32]) for k in (1,2,3))
    return result

def leaf(folder,kind,commit):
    identity=read(folder/'identity.json');assert read(folder/'status.json')['status']=='complete'
    assert identity['video_calls']==2048 and identity['unique_videos']==512 and identity['Qwen_not_executed']
    assert identity['cap']==100 and identity['nonexclusive_authorized']
    proof=read(folder/'consistency.json');assert proof['status']=='passed' and proof['serial_pipeline_all_outputs_equal'] and proof['original_first_full_field_output_equal']
    source=original_sources(folder,identity,commit);platform=guard(folder,identity)
    inputs=read(folder/'input_files.json');assert len(inputs)==512
    control=read(folder/'control/pipeline_timing.json')
    cr=control['records'];assert len(cr)==128
    for i,row in enumerate(cr):
        assert row['field']==i and row['optical_passes']==6
        assert row['front_start']<=row['front_end']<=row['body_start']<=row['first_optic']<=row['body_end']
    ctime=cr[-1]['body_end']-cr[0]['front_start'];near(ctime,control['continuous_GPU_timeline_ms'])
    control_array=predictions(folder/'control','pipeline_outputs.npz')
    is_graph=kind in ('graph','split');child=folder/('graph' if is_graph else kind)
    optimized=read(child/('graph_timing.json' if is_graph else 'pipeline_timing.json'))
    rows=optimized['rows' if is_graph else 'records'];assert len(rows)==128
    if is_graph:
        assert optimized['eager_real_fields']==3 and optimized['eager_real_video_calls']==48
        assert optimized['graph_replays']==125 and optimized['graph_real_video_calls']==2000
        assert optimized['no_dummy_or_discarded_inferences'] and optimized['first_eager_calls_and_capture_cost_in_continuous_window']
        assert optimized['raw_inputs_copied_on_every_field'] and optimized['all_neural_features_recomputed_on_every_graph_replay']
        assert optimized['fusion_diagnostic_GPU_arithmetic_and_copy_included']
        capture=optimized['capture'];assert capture['first_replayed_field']==3 and capture['captured_original_full_field_propagation_calls']==6
        near(capture['host_ms'],(capture['host_end']-capture['host_start'])*1000)
        for i,row in enumerate(rows):
            assert row['field']==i and row['kind']==('eager_real' if i<3 else 'graph_real')
            assert row['start_ms']<=row['end_ms']
            if kind=='split':
                assert row['start_ms']<=row['front_end_ms']<=row['body_start_ms']<=row['end_ms']
                if i:assert row['body_start_ms']>=rows[i-1]['end_ms']-.01
                if i>=5:assert row['start_ms']>=rows[i-2]['end_ms']-.01
            elif i:assert row['start_ms']>=rows[i-1]['end_ms']-.01
        if kind=='split':
            assert optimized['separate_front_and_body_graphs'] and optimized['double_buffer_with_previous_score_completion_dependency']
            assert len(capture['captured_buffers'])==2 and capture['total_graphs']==4
            assert all(r['propagation_calls']==6 for r in capture['captured_buffers'])
            assert read(child/'AST_partition.json')['original_tail_AST_unchanged']
        elapsed=rows[-1]['end_ms']-rows[0]['start_ms']
        array=predictions(child,'graph_outputs.npz')
        with np.load(child/'graph_outputs.npz',allow_pickle=False) as f:diag=f['diagnostics'].copy()
        assert diag.shape==(128,16) and np.isfinite(diag).all()
        assert all(np.array_equal(diag[:32],diag[k*32:(k+1)*32]) for k in (1,2,3))
        eq=read(child/'full_output_equivalence.json')
        assert eq['original_last_full_tree_vs_graph_bitwise_equal'] and eq['all_16_last_field_diagnostics_equal_original'] and eq['all_fields_diagnostic_tensors_saved']
    else:
        for i,row in enumerate(rows):
            assert row['field']==i and row['optical_passes']==6
            assert row['front_start']<=row['front_end']<=row['body_start']<=row['first_optic']<=row['body_end']
        elapsed=rows[-1]['body_end']-rows[0]['front_start']
        array=predictions(child,'pipeline_outputs.npz');capture=None
        eq=read(child/'full_output_equivalence.json')
        assert eq['original_vs_partitioned_full_tree_bitwise_equal']
        partition=read(child/'AST_partition.json');assert partition
    assert eq['status']=='passed' and eq['validation_after_timing']
    assert np.array_equal(control_array,array)
    near(elapsed,optimized['continuous_GPU_timeline_ms']);near(elapsed/128,optimized['time16_ms'])
    near(ctime/128,control['time16_ms'])
    token=read(child/'online_tokenizer.json');tr=token['records'];assert len(tr)==128
    reference=read(folder/'external_preparation.json')['rows'][0]
    for i,row in enumerate(tr):
        assert row['field']==i and row['input_ids']==reference['input_ids'] and row['mask']==reference['mask'] and row['end']>=row['start']
    near(token['CPU_work_sum_ms'],sum((r['end']-r['start'])*1000 for r in tr))
    prefixes=[]
    start=rows[0]['start_ms' if is_graph else 'front_start']
    for p in optimized['prefixes']:
        end=rows[p['video_calls']//16-1]['end_ms' if is_graph else 'body_end']
        near(p['time16_ms'],(end-start)*16/p['video_calls']);prefixes.append(p)
    measured_overlap=None
    if kind=='split':
        front=sum(r['front_end_ms']-r['start_ms'] for r in rows)
        crossed=sum(max(0,min(a['front_end_ms'],b['end_ms'])-max(a['start_ms'],b['body_start_ms'])) for i,b in enumerate(rows) for a in rows[i+1:])
        measured_overlap=dict(later_front_with_current_body_envelope_ms=crossed,front_envelope_sum_ms=front,
                              fraction=crossed/front,not_kernel_concurrency_or_physical_optics_coverage=True)
    return dict(kind=kind,repeat=identity['repeat'],video_calls=2048,fields=128,frames_per_field=64,
                control_GPU_total_ms=ctime,optimized_GPU_total_ms=elapsed,
                control_host_total_ms=control['dispatch_to_last_score_host_ms'],
                optimized_host_total_ms=optimized['dispatch_to_last_score_host_ms'],
                actual_CPU_tokenizer_work_ms=token['CPU_work_sum_ms'],capture=capture,prefixes=prefixes,measured_overlap=measured_overlap,
                source=source,platform=platform,original_path=str((child/('graph_timing.json' if is_graph else 'pipeline_timing.json')).resolve())),inputs,array

def main():
    p=argparse.ArgumentParser();p.add_argument('--graph-root',type=Path,required=True);p.add_argument('--graph-commit',required=True);p.add_argument('--split-root',type=Path);p.add_argument('--split-commit');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    files=[];points=[];reference_inputs=None
    plan=[('hoisted',ROOT/'a100_lgvq_hoisted_pipeline_20261026_01','2133a93ea8214e23b4e9d1125c55e23d6e8e7b39'),('graph',args.graph_root,args.graph_commit)]
    if args.split_root:plan.append(('split',args.split_root,args.split_commit))
    previous=None
    for kind,root,commit in plan:
        files+=sealed(root,'complete','backup_receipt_complete01.json')
        for repeat in (1,2,3):
            result,inputs,outputs=leaf(root/f'lg_repeat{repeat:02d}',kind,commit);points.append(result)
            if reference_inputs is None:reference_inputs=inputs
            else:assert reference_inputs==inputs
            if previous is not None:assert np.array_equal(previous,outputs)
            previous=outputs
    for n in (1,2):files+=sealed(ROOT/f'a100_lgvq_graph_pipeline_20261026_{n:02d}','stopped','backup_receipt_capture_stop01.json')
    # Reuse the sealed Qwen review rather than running Qwen or its model again.
    old=HERE/'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/strict_serial_pipeline_20261026_01/independent_results.json'
    q=read(old);assert q['status']=='passed';qpoints=[r for r in q['all_points'] if r['system']=='Qwen']
    assert len(qpoints)==3 and all(r['strict_single_video_CPU_GPU_serial'] for r in qpoints)
    for r in qpoints:
        path=Path(r['original_path']);original=next(f for f in q['source_files'] if f['path']==str(path))
        assert path.stat().st_size==original['bytes'] and sha(path)==original['sha256'];files.append(record(path))
    for row in q['source_files']:
        if 'a100_qwen_shared_serial_20261026_' in row['path']:
            path=Path(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];files.append(record(path))
    qinputs=read(Path(qpoints[0]['original_path']).parent/'input_files.json');assert qinputs==reference_inputs
    files.append(record(old));unique={f['path']:f for f in files}
    result=dict(status='passed',review='offline SHA, raw intervals, every output and field/tokenizer count; no GPU/model execution',
                all_points=points,Qwen_retained=qpoints,Qwen_GPU_model_window_sum16_ms=q['Qwen_GPU_model_window_sum16_ms'],
                source_files=list(unique.values()),scope_warning='LG raw inputs resident; Qwen CPU prepares each video serially. Shared GPU at different times. Full continuous ratios are operational references, not fair symmetric file-to-file E2E speedup.',
                all_rounds_retained=True,no_100x_target_used_for_selection=True,physical_optical_E2E_measured=False)
    target=args.output/'independent_results.json';target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(status='passed',points=len(points),files=len(unique),output=str(target),sha256=sha(target))))

if __name__=='__main__':main()
