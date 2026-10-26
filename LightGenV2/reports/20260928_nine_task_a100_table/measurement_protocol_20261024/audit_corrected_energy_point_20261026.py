"""Small local raw replay for a completed new-scope point; no model or CUDA."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import rental_energy_triples_20261026 as account


def audit(folder, output):
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8-sig'))
    timing=read('timing.json');cost=read('cost_report.json');identity=read('identity.json')
    if read('status.json')['status']!='complete' or cost['status']!='complete':
        raise ValueError('Only fully complete points')
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    with (folder/'telemetry.csv').open(encoding='utf-8-sig',newline='') as stream:
        samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],
            float(r['watts']),float(r['utilization_percent']),float(r['memory_used_mib'])) for r in csv.DictReader(stream)]
    value=account.summarize(timing,samples,913408)
    for key in ('elapsed_s','energy_j','GPU_energy_j','time16_ms','equivalent_frame_ms','energy16_j',
                'energy_per_video_j','energy_per_frame_j','common_tokens_per_s','native_tokens_per_s'):
        if cost[key] is not None and not math.isclose(cost[key],value[key],rel_tol=1e-9,abs_tol=1e-5):
            raise ValueError('Original report disagrees: '+key)
    rows=timing['records'];n=len(rows)
    if n!=2048 or cost['cap'] not in (50,100) or identity['visible_multiprocessors']!=(54 if cost['cap']==50 else 108):
        raise ValueError('Actual count or MPS allocation differs')
    with np.load(folder/'outputs.npz',allow_pickle=False) as data:prediction=data['prediction']
    if prediction.shape!=(2048,) or not np.array_equal(prediction,np.asarray([r['prediction'] for r in rows],np.float32)):
        raise ValueError('Full original output differs from raw call records')
    if not all(np.array_equal(prediction[:512],prediction[k:k+512]) for k in (512,1024,1536)):
        raise ValueError('Four prediction cycles differ')
    if not all(r['sample_id']==rows[i%512]['sample_id'] and r['frame_positions']==rows[i%512]['frame_positions'] for i,r in enumerate(rows)):
        raise ValueError('Source video/frames differ between cycles')
    firstTF=all('first_TF_host' in r for r in rows)
    token=sum(account.original.union_ms(r['tokenizer_intervals']) for r in rows)
    stages=dict(CPU_image_preparation_excluded_ms=sum((r['CPU_end']-r['CPU_start'])*1000 for r in rows)-token,
        CPU_tokenizer_retained_ms=token,H2D_excluded_ms=sum((r['H2D_end']-r['CPU_end'])*1000 for r in rows),
        entry_dispatch_retained_ms=sum((r['GPU_host_start']-r['H2D_end'])*1000 for r in rows),
        full_GPU_model_to_score_host_ms=sum((r['GPU_ready_host']-r['GPU_host_start'])*1000 for r in rows),
        intermediate_CPU_output_retained_ms=sum((r['CPU_result_return']-r['GPU_ready_host'])*1000 for r in rows[:-1]),
        intervideo_dispatch_retained_ms=sum((b['CPU_start']-a['CPU_result_return'])*1000 for a,b in zip(rows,rows[1:])),
        final_CPU_output_outside_primary_ms=(rows[-1]['CPU_result_return']-rows[-1]['GPU_ready_host'])*1000)
    charged=sum(stages[k] for k in ('CPU_tokenizer_retained_ms','entry_dispatch_retained_ms',
        'full_GPU_model_to_score_host_ms','intermediate_CPU_output_retained_ms','intervideo_dispatch_retained_ms'))
    if not math.isclose(charged,value['elapsed_s']*1000,abs_tol=1e-4):raise ValueError('Stage partition is not exact')
    if firstTF:
        stages.update(GPU_model_entry_to_first_TF_host_ms=sum((r['first_TF_host']-r['GPU_host_start'])*1000 for r in rows),
            first_TF_to_score_host_ms=sum((r['GPU_ready_host']-r['first_TF_host'])*1000 for r in rows))
    result=dict(status='passed',model=timing['model'],cap=cost['cap'],video_calls=n,
        time16_ms=value['time16_ms'],energy16_j=value['energy16_j'],common_tokens_per_s=value['common_tokens_per_s'],
        native_tokens_per_s=None if cost['native_tokens_per_s'] is None else value['native_tokens_per_s'],
        stage_totals_ms=stages,stage_equivalent16_ms={k:v/128 for k,v in stages.items()},
        first_TF_node_recorded=firstTF,all_2048_outputs_equal_records=True,four_cycles_bitwise_equal=True,
        max_power_sample_age_s=value['max_power_sample_age_s'],energy_is_same_time_intervals=True,
        raw_sha256={name:account.original.sha(folder/name) for name in ('timing.json','telemetry.csv','outputs.npz','cost_report.json')})
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--output',type=Path);args=p.parse_args();folder=args.folder.resolve()
    print(json.dumps(audit(folder,args.output or folder/'local_independent_energy_audit01.json'),ensure_ascii=False))
