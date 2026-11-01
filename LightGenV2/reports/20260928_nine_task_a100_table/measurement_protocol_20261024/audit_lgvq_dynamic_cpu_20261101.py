"""CPU-only once-per-triple raw replay, independent FIFO and power checks."""
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from collections import defaultdict

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
OUT=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/LG_dynamic_FIFO16_20261101_01'
if os.name=='nt':OUT=Path('\\\\?\\'+str(OUT.resolve()))


def read(p):return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def close(a,b):
    if not math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-9):raise ValueError((a,b))


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)


def main():
    target=OUT/'independent_CPU_audit01.json'
    if target.exists():
        print(json.dumps(dict(status='completed_once_only_audit_reused',sha256=sha(target))));return
    import numpy as np
    import lgvq_compute_scope_20261029 as scope
    import lgvq_dynamic_cpu_20261101 as policy
    import lgvq_async_worker_20261026 as original
    from collect_lgvq_reorganized_20261030 import energy
    receipt=read(OUT/'local_raw_receipt01.json');root=OUT/'raw'
    assert receipt['suite']['status']=='complete' and receipt['suite']['completed']==[f'lg_repeat{i:02d}' for i in (1,2,3)]
    assert receipt['controller_state'] in ('Z','exited') and receipt['worker_state'] in ('Z','exited') and receipt['suite']['mps_released']
    for item in receipt['files']:
        p=root/item['name'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    frozen=read(root/'source_receipt.json')
    for name,digest in frozen['sources'].items():
        blob=subprocess.check_output(['git','show',frozen['published_commit']+':'+name.replace('\\','/')],cwd=REPO,timeout=15)
        assert hashlib.sha256(blob).hexdigest()==digest
    reference=REPO/original.REFERENCE_RELATIVE
    assert sha(reference)==original.REFERENCE_SHA256
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction'][:32].copy()
    rounds=[];all_groups=[];timelines=[];files=[]
    for repeat in (1,2,3):
        leaf=root/f'lg_repeat{repeat:02d}';report=read(leaf/'compute_report.json');raw=read(leaf/'raw_operator_records.json')
        numeric=read(leaf/'numeric_audit.json');identity=read(leaf/'compute_identity.json')
        assert read(leaf/'status.json')['status']=='complete'
        assert numeric['equal_scores']==512 and numeric['maximum_absolute_difference']==0 and numeric['warm_scores_bitwise_equal']
        assert identity['explicit_extra_warm_fields']==3 and identity['visible_SM']==108 and identity['cap']==100
        assert identity['CPU_FIFO_workers']==16 and identity['SLM_hz']==1886
        with np.load(leaf/'outputs.npz',allow_pickle=False) as f:
            assert f['prediction'].tobytes()==expected.tobytes() and f['warm_prediction'].tobytes()==expected[:3].tobytes()
        for name,key in [('raw_kernel_trace.json','raw_trace_sha256'),('raw_operator_records.json','raw_operator_records_sha256'),('outputs.npz','outputs_sha256')]:
            assert sha(leaf/name)==report[key]
        kernels=scope.traced_kernels(read(leaf/'raw_kernel_trace.json'))
        assert kernels==read(leaf/'kernel_phase_records.json')
        replay=policy.summarize_compute(kernels,raw['patch'],raw['tokenizer'],32);del kernels
        assert replay['schedule']==report['schedule']
        nodes=report['schedule']['nodes'];by={n['id']:n for n in nodes};resources=defaultdict(list)
        for n in nodes:
            close(n['end_s']-n['start_s'],n['duration_s'])
            assert all(by[p]['end_s']<=n['start_s']+1e-9 for p in n['parents'])
            resources[n['resource']].append(n)
        for group in resources.values():
            values=sorted(group,key=lambda x:x['start_s'])
            assert all(a['end_s']<=b['start_s']+1e-9 for a,b in zip(values,values[1:]))
        # Independent list-scheduling oracle. No policy helper or old thread IDs.
        workers=[(0.,i) for i in range(16)];heapq.heapify(workers)
        rows=sorted(raw['patch'],key=lambda r:(r['field'],r['slot']))
        assert len(rows)==512 and len(raw['tokenizer'])==32
        inputs=read(leaf/'input_preparation.json')['rows'];ids=[x['sample_id'] for x in inputs]
        assert [r['sample_id'] for r in rows]==ids and len(set(ids))==512
        assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()=='48b7fea193abc07f48bec9b091db0351b949387c9b44fbd228399ce3a03b655a'
        for r in rows:
            assert r['raw_patch_bytes_equal'] and r['resize_pixels_byte_equal'] and r['normalized_pixels_byte_equal'] and r['original_post_cast_BF16_bytes_equal']
            close(sum(r[k] for k in ('resize_cpu_s','normalization_cpu_s','patch_cpu_s')),r['thread_cpu_s'])
            at,w=heapq.heappop(workers);key=f'f{r["field"]}.patch{r["slot"]}'
            for suffix,phase in (('.resize','resize'),('.normalization','normalization'),('','patch')):
                n=by[key+suffix];assert n['resource']==f'cpu_dynamic_{w}'
                close(n['start_s'],at);close(n['duration_s'],r[phase+'_cpu_s']);at+=r[phase+'_cpu_s']
            heapq.heappush(workers,(at,w))
        actual_workers=defaultdict(list)
        origin=min(r['host_start'] for r in rows)
        actual=[]
        for r in rows:
            actual_workers[r['worker_thread']].append(r)
            actual.append(dict(field=r['field']+1,slot=r['slot']+1,worker=r['worker_native_id'],
                start_ms=(r['host_start']-origin)*1000,resize_end_ms=(r['resize_end_wall']-origin)*1000,
                normalize_end_ms=(r['normalization_end_wall']-origin)*1000,end_ms=(r['host_end']-origin)*1000,
                resize_cpu_ms=r['resize_cpu_s']*1000,normalization_cpu_ms=r['normalization_cpu_s']*1000,patch_cpu_ms=r['patch_cpu_s']*1000))
        assert len(actual_workers)==16
        real_gaps=[]
        for group in actual_workers.values():
            ordered=sorted(group,key=lambda r:r['host_start'])
            for a,b in zip(ordered,ordered[1:]):
                assert b['host_start']>=a['host_end'];real_gaps.append(b['host_start']-a['host_end'])
        optic=[n for n in nodes if n['resource']=='optical']
        assert len(optic)==192
        assert all(math.isclose(n['duration_s'],1/1886+.000300+.0000307,abs_tol=1e-15) for n in optic)
        close(report['time16_s'],report['schedule']['total_s']/32)
        close(sum(report['equivalent_s_per_group'].values()),report['time16_s'])
        with (leaf/'telemetry.csv').open(newline='',encoding='utf-8-sig') as f:samples=list(csv.DictReader(f))
        stamps=[float(x['host_monotonic_s']) for x in samples];watts=[float(x['watts']) for x in samples]
        assert all(math.isfinite(x) for x in stamps+watts) and all(b>a for a,b in zip(stamps,stamps[1:])) and all(0<w<=1000 for w in watts)
        start,end=raw['formal_monotonic_start'],raw['formal_monotonic_end'];assert stamps[0]<=start<end<=stamps[-1]
        segments=[(max(a,start),min(b,end),w,a) for a,b,w in zip(stamps,stamps[1:],watts) if min(b,end)>max(a,start)]
        elapsed=sum(b-a for a,b,w,t in segments);joules=sum((b-a)*w for a,b,w,t in segments)
        close(elapsed,end-start);close(joules,report['diagnostic_FFT_board_energy']['energy_j'])
        wall=raw['formal_host_end']-raw['formal_host_start'];close(wall,report['actual_FFT_software_wall_s'])
        audit=read(leaf/'process_audit.json');assert audit and all(not a['foreign_gpu_processes'] and not a['foreign_mps_clients'] for a in audit)
        assert not report['physical_hardware_measured'] and not report['energy_qualified']
        e=energy(report['schedule'],32)
        opt_intervals=[(n['start_s'],n['end_s']) for n in optic]
        ele_intervals=[(n['start_s'],n['end_s']) for n in nodes if n['resource']!='optical']
        independent_e=41.388*union(opt_intervals+ele_intervals)+62.842*union(ele_intervals)+80.358*union(opt_intervals)
        close(independent_e,e['energy_j'])
        work=defaultdict(float)
        for n in nodes:work[n['phase']]+=n['duration_s']/32
        groups=[];completed_prefix=0.
        for f in range(32):
            cpu=[n for n in nodes if n['field']==f and n['resource'].startswith('cpu_dynamic_')]
            g=dict(round=repeat,group=f+1,CPU_start_ms=min(n['start_s'] for n in cpu)*1000,
                CPU_ready_ms=max(n['end_s'] for n in cpu)*1000,completion_ms=by[f'f{f}.decoder']['end_s']*1000)
            completed_prefix=max(completed_prefix,g['completion_ms'])
            g['cumulative_ms16']=completed_prefix/(f+1)
            g['actual_CPU_ready_ms']=max(r['host_end']-origin for r in rows if r['field']==f)*1000
            g['actual_FFT_completion_ms']=next(r['host_end']-origin for r in raw['host_records'] if not r['warm'] and r['field']==f)*1000
            groups.append(g)
        all_groups+=groups
        timelines.append(dict(round=repeat,actual_CPU=actual,projected_nodes=[dict(id=n['id'],field=n['field']+1,phase=n['phase'],resource=n['resource'],start_ms=n['start_s']*1000,end_ms=n['end_s']*1000) for n in nodes],groups=groups))
        rounds.append(dict(fields=32,time16_s=report['time16_s'],total_s=report['schedule']['total_s'],
            equivalent_s_per_group=report['equivalent_s_per_group'],work_s_per_group=dict(work),component_energy=e,
            actual_FFT_wall_s=wall,actual_FFT_ms16=wall/32*1000,actual_FFT_board_J16=joules/32,
            actual_FFT_board_mean_W=joules/elapsed,actual_CPU_wall_s=report['real_CPU_wall_s'],
            actual_workers=16,actual_CPU_max_interjob_gap_ms=max(real_gaps)*1000,
            projected_CPU_continuous=True,projected_CPU_finish_ms=max(t for t,w in workers)*1000,
            CPU_work_lower_bound_ms16=sum(r['thread_cpu_s'] for r in rows)/16/32*1000,
            numeric_audit=numeric,report_sha256=sha(leaf/'compute_report.json')))
        files.extend(dict(path=str(leaf/name),sha256=sha(leaf/name)) for name in ('compute_report.json','raw_operator_records.json','numeric_audit.json'))
    baseline=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/LGVQ_standard1886_20261030_01/lgvq_reorganized_sources01.json'
    assert sha(baseline)=='7c7501e71716a0aa9e28288c7c7141848bc6fc383ad7c7920ea1a5f6196cecd9'
    old=read(baseline);phases=set().union(*(r['work_s_per_group'] for r in rounds))
    lg=dict(fields=96,rounds=3,time16_s=sum(r['total_s'] for r in rounds)/96,
        component_energy16_j=sum(r['component_energy']['energy_j'] for r in rounds)/96,
        equivalent_s_per_group={k:sum(r['equivalent_s_per_group'].get(k,0) for r in rounds)/3 for k in phases},
        work_s_per_group={k:sum(r['work_s_per_group'].get(k,0) for r in rounds)/3 for k in phases},source_rounds=rounds)
    result=dict(status='independent_CPU_dynamic_FIFO16_triple_passed',source_commit=frozen['published_commit'],
        raw_receipt_sha256=sha(OUT/'local_raw_receipt01.json'),raw_files=len(receipt['files']),
        numeric_scores_exact=1536,prospective_rounds=3,formal_fields=96,workers=16,
        modelled_ms16_mean=lg['time16_s']*1000,modelled_ms16_sample_std=statistics.stdev(r['time16_s']*1000 for r in rounds),
        modelled_component_J16=lg['component_energy16_j'],actual_FFT_ms16_mean=statistics.mean(r['actual_FFT_ms16'] for r in rounds),
        physical_optics_latency_or_energy_measured=False,all_workers_exited=True,MPS_released=True,
        Torch_imported='torch' in sys.modules,rounds=rounds)
    assert not result['Torch_imported']
    value=dict(status=result['status'],LG_lab=lg,baselines=old['baselines'],trials=old['trials'],common_tokens16=old['common_tokens16'],
        component_powers_w=old['component_powers_w'],SLM_hz=1886,includes_full_frontend_preparation=False,
        hardware=read(root/'environment.json'),source_files=files,source_commit=frozen['published_commit'],
        groups=all_groups,measurement_vs_projection=result)
    for name,data in [('independent_CPU_audit01.json',result),('workbook_sources01.json',value),('timeline_sources01.json',timelines)]:
        with (OUT/name).open('x') as f:json.dump(data,f,ensure_ascii=True,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rounds'},ensure_ascii=True))


if __name__=='__main__':main()
