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

def review_leaf(folder):
    status=read(folder/'status.json');identity=read(folder/'identity.json')
    assert status['status']=='complete'
    assert identity['video_calls']==2048 and identity['unique_videos']==512
    receipt=read(folder/'backup_receipt_complete01.json')
    files=[]
    for f in receipt['verified_files']:
        p=folder/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
        files.append(dict(path=str(p.resolve()),bytes=f['bytes'],sha256=f['sha256']))
    for p,digest in identity['sources'].items():
        # The running server HEAD was intentionally not advanced across active
        # peer training. Resolve each actual source against the published commit.
        blob=subprocess.check_output(['git','show',identity['published_measurement_commit']+':'+p],cwd=REPO)
        assert hashlib.sha256(blob).hexdigest()==digest,p
    guard=read(folder/'process_audit.json');assert guard
    pid=identity['pid']
    for g in guard:
        assert set(g['clients'])<={pid}
        assert all(int(x[0].strip()) in set(g['servers'])|{pid} for x in g['processes'])
        assert g['platform']['power_limit_w']==250 and g['platform']['uuid']==identity['gpu_uuid']
    inputs=read(folder/'input_files.json');assert len(inputs)==512 and len({x['sample_id'] for x in inputs})==512
    prep=read(folder/'external_preparation.json')
    system=identity['system'];out=[]
    if system=='lg':
        consistency=read(folder/'consistency.json');assert consistency['status']=='passed'
        assert consistency['serial_pipeline_all_outputs_equal'] and consistency['original_first_full_field_output_equal']
        first_array=None
        for mode in ('serial','pipeline'):
            d=read(folder/(mode+'_timing.json'));records=d['records'];assert len(records)==128
            begin=records[0]['front_start'];end=records[-1]['body_end'];elapsed=end-begin
            near(elapsed,d['continuous_GPU_timeline_ms']);near(elapsed/128,d['time16_ms'])
            front_intervals=[];body_intervals=[];stage_totals=defaultdict(float);before_first_optic=0.;after_first_optic=0.;extra_GPU=0.
            for i,r in enumerate(records):
                assert r['field']==i and r['front_start']<=r['front_end']<=r['body_start']<=r['body_end']
                if i:
                    assert r['front_start']>=records[i-1]['front_end']-.001
                    assert r['body_start']>=records[i-1]['body_end']-.001
                front_intervals.append((r['front_start'],r['front_end']));body_intervals.append((r['body_start'],r['body_end']))
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
            assert steady_report(starts)==d['steady']
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
                stages_per16_ms={k:v/128 for k,v in stage_totals.items()},
                theoretical_only=dict(P_GPU_before_first_optics_ms=P,E_after_first_optics_ms=E,O_fixed_ms=O,
                    serial_segment_projection16_ms=P+E+O,independent_resources_cycle_ms=max(P,E+O),
                    warning='Segment sum has incomplete scope coverage and mixed-stream envelopes. Not measured physical E2E or verified hardware overlap.'),
                original_path=str((folder/(mode+'_timing.json')).resolve())))
    else:
        d=read(folder/'qwen_timing.json');records=d['records'];assert len(records)==2048
        elapsed=records[-1]['end_ms']-records[0]['start_ms'];near(elapsed,d['continuous_GPU_timeline_ms'])
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
        assert steady_report(groupends)==d['steady']
        pre=sum(r['first_TF_ms']-r['start_ms'] for r in records)/128
        post=sum(r['end_ms']-r['first_TF_ms'] for r in records)/128
        out.append(dict(system='Qwen',mode='native_serial',repeat=identity['repeat'],cap=50,video_calls=2048,
            elapsed_ms=elapsed,time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,
            model_front16_ms=pre,first_TF_to_score16_ms=post,inter_video_gaps16_ms=gaps/128,
            first_qualified_steady_videos=d['steady']['first_qualified_prefix_videos'],prefixes=d['prefixes'],
            original_path=str((folder/'qwen_timing.json').resolve())))
    tokenize=prep.get('tokenizer_calls',prep.get('tokenizer',[]))
    extra=dict(system=system,repeat=identity['repeat'],recorded_CPU_tokenizer_ms=sum((r['end']-r['start'])*1000 for r in tokenize),
        tokenizer_call_count=len(tokenize),prepared_unique_inputs=512,preparation_reused_as_raw_input_across4cycles=True,
        wall_note='External CPU preparation and H2D were separately observed once per512 raw inputs, not part of GPU-input-resident primary timeline.')
    return out,inputs,files,extra


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    points=[];sources=[];inputs_ref=None;extras=[]
    for system in ('lg','qwen'):
        for rep in (1,2,3):
            pts,inputs,files,extra=review_leaf(a.root/f'{system}_repeat{rep:02d}');points+=pts;sources+=files;extras.append(extra)
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
    result=dict(status='passed',matched_video_calls_per_repeat=2048,unique_original_videos=512,cycles=4,
        accuracy_not_re_evaluated=True,physical_optical_timing_not_measured=True,
        boundary='raw model inputs already GPU-resident; first GPU frontend through last score on GPU; all internal waits retained',
        all_points=points,aggregates=aggregate,extra_preparation=extras,
        actual_input_hash_sequence_verified=True,measured_overlap_is_software_body_not_physical_6cycle_coverage=True,
        source_files=sources)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',aggregates=aggregate,output=str(a.output)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
