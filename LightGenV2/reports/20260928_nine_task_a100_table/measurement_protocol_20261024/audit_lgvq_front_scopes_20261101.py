"""Once-only independent CPU audit of the four completed prospective triples.

No model imports, GPU work, or raw-file edits. Rebuild device interval unions,
stage means, formula closure, and resource/dependency checks from saved data.
"""
import hashlib
import csv
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/LG_four_front_scopes_20261101_01'
COMMIT='151c034c506073d042b5cc77b24a637ca38141bb'
CPU=('decode','RGB','crop','resize','normalization','patch')


def read(p):return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10):raise ValueError((a,b))


def union(intervals):
    total=0.;previous=None
    for start,end in sorted(intervals):
        assert end>=start
        total+=end-start if previous is None else max(0.,end-max(start,previous))
        previous=end if previous is None else max(previous,end)
    return total/1e6


def main():
    target=OUT/'independent_CPU_audit01.json'
    if target.exists():raise ValueError('Accepted audit exists; do not repeat')
    receipt=read(OUT/'local_raw_receipt01.json')
    assert receipt['suite']['status']=='complete' and receipt['suite']['mps_released']
    assert receipt['controller_state'] in ('Z','exited')
    assert receipt['suite']['source_commit']==COMMIT
    expected=[f'v{v}_repeat{i:02d}' for v in range(4) for i in (1,2,3)]
    assert receipt['suite']['completed']==expected
    for item in receipt['files']:
        p=OUT/'raw'/item['name'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    import numpy as np
    old=OUT.parent/'LG_dynamic_FIFO16_20261101_01'
    old_receipt=read(old/'local_raw_receipt01.json')
    old_output=old/'raw/lg_repeat01/outputs.npz'
    pinned=next(r['sha256'] for r in old_receipt['files'] if r['name']=='lg_repeat01/outputs.npz')
    assert sha(old_output)==pinned
    with np.load(old_output,allow_pickle=False) as f:
        expected_prediction=f['prediction'].copy();expected_warm=f['warm_prediction'].copy()
    rounds=[];group_rows=[]
    for v in range(4):
        for repeat in (1,2,3):
            name=f'v{v}_repeat{repeat:02d}';leaf=OUT/'raw'/name
            derived=name=='v1_repeat01' and (leaf/'CPU_derivation_receipt01.json').exists()
            report_file='derived_compute_report01.json' if derived else 'compute_report.json'
            report=read(leaf/report_file);raw=read(leaf/'raw_operator_records.json')
            kernels=read(leaf/('derived_kernel_phase_records01.json' if derived else 'kernel_phase_records.json'))
            numeric=read(leaf/('derived_numeric_audit01.json' if derived else 'numeric_audit.json'))
            identity=read(leaf/'compute_identity.json');status=read(leaf/'status.json')
            parent=read(leaf/'parent_CPU_admission.json')
            assert identity['explicit_extra_warm_fields']==3 and identity['fields']==32
            assert identity['variant']==v and identity['SLM_hz']==1886
            assert numeric['all_formal_scores_bitwise_equal'] and numeric['warm_scores_bitwise_equal']
            assert numeric['equal_scores']==512 and numeric['maximum_absolute_difference']==0
            if derived:
                derived_receipt=read(leaf/'CPU_derivation_receipt01.json')
                assert status['status']=='failed' and derived_receipt['status']=='complete_separate_CPU_derivation'
                assert derived_receipt['no_invalid_power_inside_formal_window'] and derived_receipt['no_Torch_or_GPU']
                assert derived_receipt['report_sha256']==sha(leaf/report_file)
                assert all(sha(leaf/name)==digest for name,digest in derived_receipt['original_files_before_SHA256'].items())
            else:
                assert status['status']=='complete' and parent['status']=='passed'
            assert parent['report_sha256']==sha(leaf/report_file)
            with np.load(leaf/'outputs.npz',allow_pickle=False) as f:
                for key,expected_array in [('prediction',expected_prediction),('warm_prediction',expected_warm)]:
                    actual=f[key];assert actual.shape==expected_array.shape and actual.dtype==expected_array.dtype and actual.tobytes()==expected_array.tobytes()
            rows=raw['patch'];assert len(rows)==512 and len(raw['host_records'])==35
            assert {(r['field'],r['slot']) for r in rows}=={(f,s) for f in range(32) for s in range(16)}
            assert len({r['sample_id'] for r in rows})==512 and all(r['raw_patch_bytes_equal'] for r in rows)
            assert sum(r['warm'] for r in raw['host_records'])==3
            grouped={}
            for k in kernels:
                if 0<=k['field']<32:grouped.setdefault((k['field'],k['phase']),[]).append((k['start_us'],k['end_us']))
            device={key:union(values) for key,values in grouped.items()}
            phase_means={}
            for p in {p['phase'] for p in report['raw_phases']}:
                values=[device[(f,p)] for f in range(32)]
                phase_means[p]=1/1886+.000300+.0000307 if p.startswith('optics') else statistics.mean(values)
            tokenizer=statistics.mean(r['thread_cpu_s'] for r in raw['tokenizer'])
            selected=CPU[3:] if v==0 else CPU[:v]+CPU[3:]
            cpu={p:sum(r.get(p+'_cpu_s',0.) for r in rows)/512 for p in CPU}
            P=sum(cpu[p] for p in selected)
            Q=sum(value for p,value in phase_means.items() if p!='emb_text')
            mean=report['mean_stage_pipeline']
            assert mean['load_imbalance_charged'] is False and mean['simple_P_plus_Q_over_N_verified']
            close(P,mean['CPU_mean_s_per_field']);close(Q,mean['mean_backend_after_image_s'])
            close(P+Q/32,report['time16_s']);close(Q/32,mean['tail_amortized_s_per_field'])
            for p in phase_means:close(phase_means[p],mean['backend_phase_mean_s_per_field'][p])
            close(tokenizer,mean['backend_phase_mean_s_per_field']['tokenizer'])
            nodes=mean['schedule']['nodes'];by={n['id']:n for n in nodes};resources={}
            for n in nodes:
                close(n['end_s']-n['start_s'],n['duration_s'])
                assert all(by[p]['end_s']<=n['start_s']+1e-9 for p in n['parents'])
                resources.setdefault(n['resource'],[]).append(n)
                phase=n['phase'];close(n['duration_s'],cpu[phase] if '.CPU.' in n['id'] else tokenizer if phase=='tokenizer' else phase_means[phase])
            for resource,items in resources.items():
                items.sort(key=lambda n:n['start_s'])
                assert all(a['end_s']<=b['start_s']+1e-9 for a,b in zip(items,items[1:]))
            optics=[(n['start_s']*1e6,n['end_s']*1e6) for n in nodes if n['resource']=='optical']
            electronic=[(n['start_s']*1e6,n['end_s']*1e6) for n in nodes if n['resource']!='optical']
            controller_s=union(optics+electronic);electronic_s=union(electronic);optical_s=union(optics)
            close(controller_s,mean['schedule']['total_s'])
            energy=dict(controller_J16=41.388*controller_s/32,electronic_J16=62.842*electronic_s/32,
                        optical_J16=80.358*optical_s/32,controller_s16=controller_s/32,
                        electronic_s16=electronic_s/32,optical_s16=optical_s/32)
            energy['total_J16']=sum(energy[k] for k in ('controller_J16','electronic_J16','optical_J16'))
            with (leaf/'telemetry.csv').open(newline='',encoding='utf-8-sig') as f:samples=list(csv.DictReader(f))
            stamps=[float(x['host_monotonic_s']) for x in samples];watts=[float(x['watts']) for x in samples]
            assert all(math.isfinite(x) for x in stamps+watts) and all(b>a for a,b in zip(stamps,stamps[1:])) and all(0<w<=1000 for w in watts)
            begin,end=raw['formal_monotonic_start'],raw['formal_monotonic_end']
            assert stamps[0]<=begin<end<=stamps[-1]
            spans=[(max(a,begin),min(b,end),w) for a,b,w in zip(stamps,stamps[1:],watts) if min(b,end)>max(a,begin)]
            real_joules=sum((b-a)*w for a,b,w in spans);real_seconds=sum(b-a for a,b,w in spans)
            close(real_seconds,end-begin);close(real_joules,report['diagnostic_FFT_board_energy']['energy_j'])
            audit=read(leaf/'process_audit.json')
            assert audit and all(not r['foreign_gpu_processes'] and not r['foreign_mps_clients'] for r in audit)
            for f in range(32):
                end=by[f'f{f}.decoder']['end_s'];ready=by[f'f{f}.CPU.patch']['end_s']
                close(ready,(f+1)*P)
                group_rows.append(dict(variant=v,repeat=repeat,field=f+1,
                                       estimated_CPU_ready_ms=ready*1000,estimated_score_done_ms=end*1000,
                                       cumulative_average_ms=end/(f+1)*1000,
                                       actual_CPU_ready_ms=(max(r['host_end'] for r in rows if r['field']==f)-raw['formal_host_start'])*1000,
                                       actual_FFT_score_done_ms=(next(r['host_end'] for r in raw['host_records'] if not r['warm'] and r['field']==f)-raw['formal_host_start'])*1000))
            front={}
            for p in CPU[:3]:
                segments=[s for r in rows for s in r['frontend_measurements'] if s['phase']==p]
                active=sum(s['thread_cpu_s'] for s in segments)/512
                elapsed=sum(s['elapsed_s'] for s in segments)/512
                front[p]=dict(mean_active_ms_per_video=active*1000,
                              mean_worker_elapsed_ms_per_video=elapsed*1000,
                              mean_off_CPU_clock_difference_ms_per_video=(elapsed-active)*1000,
                              negative_difference_intervals=sum(s['off_CPU_or_clock_difference_s']<0 for s in segments),
                              measured=v!=0)
                if v:close(active,cpu[p])
            realwall=report['actual_FFT_software_wall_s']/32
            actualCPU=(max(r['host_end'] for r in rows)-min(r['host_start'] for r in rows))/32
            rounds.append(dict(variant=v,repeat=repeat,name=name,CPU_stage_mean_ms={p:cpu[p]*1000 for p in CPU},
                               backend_stage_mean_ms={p:t*1000 for p,t in phase_means.items()},tokenizer_mean_ms=tokenizer*1000,
                               CPU_mean_ms=P*1000,backend_mean_tail_ms=Q*1000,amortized_tail_ms=Q/32*1000,
                               estimated_ms16=report['time16_s']*1000,actual_CPU_queue_ms16=actualCPU*1000,
                               actual_FFT_software_ms16=realwall*1000,
                               actual_FFT_board_J16=real_joules/32,
                               actual_FFT_board_mean_W=real_joules/real_seconds,
                               component_energy_estimate=energy,
                               frontend_measurements=front,report_sha256=sha(leaf/report_file),separate_CPU_derivation=derived,
                               raw_sha256=sha(leaf/'raw_operator_records.json')))
    variants=[]
    for v in range(4):
        rs=[r for r in rounds if r['variant']==v]
        variants.append(dict(variant=v,mean_ms16=statistics.mean(r['estimated_ms16'] for r in rs),
                             sample_SD_ms16=statistics.stdev(r['estimated_ms16'] for r in rs),
                             CPU_mean_ms=statistics.mean(r['CPU_mean_ms'] for r in rs),
                             amortized_tail_ms=statistics.mean(r['amortized_tail_ms'] for r in rs),
                             component_energy_J16=statistics.mean(r['component_energy_estimate']['total_J16'] for r in rs)))
    value=dict(status='independent_CPU_four_scope_triples_passed',published_runtime_commit=COMMIT,
               audit_source_sha256=sha(Path(__file__)),
               formal_scores_bitwise_equal=6144,warm_scores_bitwise_equal=576,rounds=rounds,variants=variants,
               estimated_groups=group_rows,raw_receipt_sha256=sha(OUT/'local_raw_receipt01.json'),
               sources_verified=len(receipt['files']),Torch_imported='torch' in sys.modules,
               actual_and_ideal_estimate_separate=True,CPU_load_imbalance_not_charged=True,
               off_CPU_difference_not_claimed_as_pure_disk_wait=True)
    assert not value['Torch_imported']
    with target.open('x') as f:json.dump(value,f,ensure_ascii=True,indent=2,allow_nan=False)
    print(json.dumps(dict(status=value['status'],variants=variants)))


if __name__=='__main__':main()
