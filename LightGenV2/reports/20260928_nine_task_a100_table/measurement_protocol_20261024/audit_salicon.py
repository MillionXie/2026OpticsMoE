"""Independent CPU replay of SALICON raw maps, stages, source and energy."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from audit_t07 import REPO,sha,read,csv_rows,close

PINS={'ours':'036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe',
      'qwen':'689d086b5a390c399df109d07b73185575cd2245998b7bd427f6ccea5b0fe019'}
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def audit(root,manifest):
    state=read(root/'suite_status.json')
    if state['status']!='complete' or not state['mps_released'] or len(state['completed'])!=6:
        raise ValueError('All six fresh processes and release required')
    expected=[r for r in csv_rows(manifest) if r['split']=='validation'][:200]
    targets=root/'audit_targets';receipt=read(targets/'transfer_receipt.json')
    if [r['sample_id'] for r in receipt]!=[r['sample_id'] for r in expected]:raise ValueError('Target ID order')
    density=[]
    for record in receipt:
        path=targets/record['file']
        if sha(path)!=record['sha256']:raise ValueError('Original target SHA')
        with Image.open(path) as image:target=np.asarray(image.convert('L').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        if target.sum()<=0:raise ValueError('Empty target')
        density.append(target/target.sum())
    # Independently implement Pearson, not importing any training evaluator.
    truth=np.stack(density).astype(np.float64).reshape(200,-1)
    truth-=truth.mean(axis=1,keepdims=True)
    reports=[];sources=set();details=[]
    for method in ('ours','qwen'):
        previous=None;previous_inputs=None
        for repeat in (1,2,3):
            folder=root/f'{method}_repeat{repeat:02d}'
            info=read(folder/'identity.json');summary=read(folder/'summary.json')
            samples=csv_rows(folder/'samples.csv');stages=csv_rows(folder/'stages.csv')
            if info['checkpoint_sha256']!=PINS[method] or info['manifest_sha256']!=sha(manifest):raise ValueError('Checkpoint or manifest identity')
            if info['explicit_warmup_forwards']!=0 or info['repeat']!=repeat:raise ValueError('Wrong protocol')
            if len(samples)!=200 or [r['sample_id'] for r in samples]!=[r['sample_id'] for r in expected]:raise ValueError('200 TEST order')
            if len({r['sample_id'] for r in samples})!=200:raise ValueError('Duplicate TEST')
            for record in read(folder/'files_manifest.json'):
                path=folder/record['file']
                if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('Raw SHA/size')
            inputs=[]
            for sample,expect in zip(samples,expected):
                if sample['image_path']!=expect['image_path'] or json.loads(sample['processed_size'])!=[224,224]:raise ValueError('Wrong transform')
                if json.loads(sample['pixel_values_shape'])!=[196,1536] or json.loads(sample['image_grid_thw'])!=[[1,14,14]]:raise ValueError('Wrong native patch input')
                inputs.append((sample['sample_id'],sample['image_sha256'],sample['original_size']))
            if previous_inputs is not None and inputs!=previous_inputs:raise ValueError('Repeated input identity differs')
            previous_inputs=inputs
            prediction=np.load(folder/'predicted_density_maps.npz',allow_pickle=False)['maps']
            if prediction.shape!=(200,224,224) or not np.isfinite(prediction).all():raise ValueError('Incomplete prediction maps')
            if previous is not None and not np.array_equal(prediction,previous):raise ValueError('Repeated predictions differ')
            previous=prediction
            x=prediction.astype(np.float64).reshape(200,-1);x-=x.mean(axis=1,keepdims=True)
            denom=np.sqrt((x*x).sum(1)*(truth*truth).sum(1))
            cc=np.divide((x*truth).sum(1),denom,out=np.zeros(200),where=denom>0)
            for row,value in zip(samples,cc):close(value,float(row['cc_float64']))
            close(cc.mean(),summary['measured200_cc_float64'])
            # Published Git provenance for measurement script, config and ledger.
            script='LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/measure_salicon.py'
            bindings=[(script,info['script_sha256'])]
            bindings.append((info['config'].split('/2026OpticsMoE/',1)[1],info['config_sha256']))
            for item in read(folder/'stage_ledger.json'):
                if '/2026OpticsMoE/' in item['source_file']:bindings.append((item['source_file'].split('/2026OpticsMoE/',1)[1],item['source_sha256']))
            for name,digest in bindings:
                blob=subprocess.check_output(['git','show',info['source_commit']+':'+name],cwd=REPO)
                if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Unpublished or modified measured source')
                sources.add((name,digest))
            events=read(folder/'process_audit.json');own_pid=None
            if len(events)<2:raise ValueError('Missing occupation audit')
            for event in events:
                if len(event['clients'])!=1:raise ValueError('Foreign/missing client')
                if own_pid is None:own_pid=event['clients'][0]
                if event['clients']!=[own_pid]:raise ValueError('Client changed')
                for proc in event['processes']:
                    if int(proc[0]) not in {own_pid,*event['servers']} or proc[1].strip()!=GPU:raise ValueError('Foreign compute')
            bysample=defaultdict(list);counts=Counter();stage_total=defaultdict(float);charged=0.;nn_wall=0.;nn_event=0.
            for row in stages:
                index=int(row['sample_index']);start=float(row['start_monotonic']);end=float(row['end_monotonic'])
                if index not in range(200) or start>end:raise ValueError('Stage order/index')
                close((end-start)*1000,float(row['synchronized_wall_ms']))
                if start<float(samples[index]['input_start']) or end>float(samples[index]['cpu_result_return']):raise ValueError('Stage outside inference')
                bysample[index].append((start,end));counts[row['stage']]+=1;stage_total[row['stage']]+=float(row['synchronized_wall_ms'])
                if row['charged']=='True':
                    charged+=float(row['synchronized_wall_ms'])
                    if row['cuda_event_ms']:
                        nn_wall+=float(row['synchronized_wall_ms']);nn_event+=float(row['cuda_event_ms'])
            for spans in bysample.values():
                for a,b in zip(spans,spans[1:]):
                    if a[1]>b[0]+1e-8:raise ValueError('Overlapping charged/audit spans')
            gap=0.
            for i,row in enumerate(samples):
                actual=0. if i==0 else (float(row['input_start'])-float(samples[i-1]['cpu_result_return']))*1000
                close(actual,float(row['internal_gap_ms']));gap+=actual
            charged+=gap
            seconds=float(samples[-1]['cpu_result_return'])-float(samples[0]['input_start'])
            close(seconds,summary['actual_software_wall_seconds'])
            if method=='ours':
                for name,n in [('bridge.native_merger',200),('vision.router_simulation',200),('vision.expert_global_simulation',400),('vision.parallel_residual1',200),('vision.parallel_residual2',200),('head.saliency_decoder',200)]:
                    if counts[name]!=n:raise ValueError('Missing native merger, three passes, residual or head')
                for row in stages:
                    if row['family'] in ('simulation_only','parallel_residual','fusion','layout') and row['charged']!='False':raise ValueError('Excluded arithmetic charged')
                optical=200*3*1.0447;duration=(charged+optical)/1000
                energy=(121.746*optical+104.230*charged)/1000
                close(charged,summary['charged_electronic_ms_total']);close(optical,summary['optical_ms_total']);board_mean=None
            else:
                if counts['qwen.full_native_vision_aligned_head']!=200:raise ValueError('Full native baseline not timed')
                power=csv_rows(folder/'telemetry.csv');ts=[float(r['host_monotonic_s']) for r in power]
                start=float(samples[0]['input_start']);end=float(samples[-1]['cpu_result_return'])
                if ts[0]>start or ts[-1]<end or any(b<=a for a,b in zip(ts,ts[1:])):raise ValueError('Power bracketing/order')
                tick=start;j=bisect_right(ts,start)-1;board=0.;max_age=0.
                while tick<end:
                    boundary=min(end,ts[j+1] if j+1<len(ts) else end);age=boundary-ts[j]
                    if age>.25:raise ValueError('Power support older than250ms')
                    board+=float(power[j]['watts'])*(boundary-tick);max_age=max(max_age,age);tick=boundary;j+=1
                duration=seconds;energy=338.2*seconds+board;optical=0.;charged=1000*seconds;board_mean=board/seconds
                close(board_mean,summary['mean_board_power_w']);close(energy,summary['power_integration']['energy_j'])
            close(duration*5,summary['cost']['sample_ms']);close(energy/200,summary['cost']['sample_j'])
            reports.append(dict(system=method,repeat=repeat,sample_ms=duration*5,sample_j=energy/200,
                                charged_electronic_ms=charged/200,optical_ms=optical/200,subset200_cc=float(cc.mean()),
                                observed_board_w=board_mean,nn_wall_event_ratio=nn_wall/nn_event,summary_sha256=sha(folder/'summary.json')))
            for name in counts:
                example=next(r for r in stages if r['stage']==name)
                details.append(dict(system=method,repeat=repeat,stage=name,family=example['family'],charged=example['charged'],occurrences=counts[name],ms_per_sample=stage_total[name]/200))
    aggregate={}
    for method in ('ours','qwen'):
        rows=[r for r in reports if r['system']==method]
        aggregate[method]={key:sum(r[key] for r in rows)/3 for key in ('sample_ms','sample_j','charged_electronic_ms','optical_ms','subset200_cc')}
        if method=='qwen':aggregate[method]['time_weighted_board_w']=sum(r['observed_board_w']*r['sample_ms'] for r in rows)/sum(r['sample_ms'] for r in rows)
    return dict(status='passed',calls_per_system=600,unique_test_images=200,original_full_test_images=5000,
                repeats=reports,aggregate=aggregate,source_bindings=sorted(sources),stage_details=details,
                tokens_reference='196 native input patches/image, vision-only, not LLM generation tokens',
                op_count_status='pending actual matched input replay; not copied from historical profiles')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.root,args.manifest)
    with args.output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('status','calls_per_system','aggregate')}))
