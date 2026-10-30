"""Requested LG-first table sources. CPU-only saved-trace calculation, no rerun."""
import argparse,hashlib,json,math,subprocess,sys
from pathlib import Path
import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as prep
from lgvq_async_component_energy_20261026 import component_energy

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
SIM=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
OUT=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b'
DEST=OUT/'LGVQ_LG_first_parallel_20261030_01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8):raise ValueError((a,b))

def energy(schedule,fields):
    # Same previously approved component powers and interval UNION accounting.
    nodes=[dict(id=n['id'],kind='optical_simulation' if n['resource']=='optical' else 'electronics') for n in schedule['nodes']]
    clock=dict(starts_ms={n['id']:n['start_s']*1000 for n in schedule['nodes']},ends_ms={n['id']:n['end_s']*1000 for n in schedule['nodes']},wall_ms=schedule['total_s']*1000)
    return component_energy(nodes,clock,fields)

def entry(report,raw,*,rental=False,slm_hz=1400):
    if slm_hz!=1400:
        from recalculate_lg_front1886_20261030 import nodes_for
        schedule=scope.compute_schedule(nodes_for(report,raw,slm_hz))
        by={n['id']:n for n in schedule['nodes']};parts={}
        for name in schedule['additive_path']:
            n=by[name];parts[n['phase']]=parts.get(n['phase'],0)+n['duration_s']/report['fields']
        report=dict(report,schedule=schedule,time16_s=schedule['total_s']/report['fields'],equivalent_s_per_group=parts)
    schedule=report['schedule'];fields=report['fields']
    close(sum(report['equivalent_s_per_group'].values()),schedule['total_s']/fields)
    close(report['time16_s'],schedule['total_s']/fields)
    if fields!=32 or len(raw['patch'])!=512 or len(raw['tokenizer'])!=32:raise ValueError('Original complete fields required')
    native=sum(sum(sum(x) for x in r['mask']) for r in raw['tokenizer'])
    work={}
    for n in schedule['nodes']:work[n['phase']]=work.get(n['phase'],0)+n['duration_s']/fields
    e=energy(schedule,fields)
    return dict(fields=fields,time16_s=report['time16_s'],total_s=schedule['total_s'],native_tokens=native,
                equivalent_s_per_group=report['equivalent_s_per_group'],work_s_per_group=work,
                component_energy=e,rental_numeric_gate_passed=not rental)

def collect(standard1886=False):
    dest=OUT/'LGVQ_standard1886_20261030_01' if standard1886 else DEST
    slm_hz=1886 if standard1886 else 1400
    file=dest/'lgvq_reorganized_sources01.json'
    if file.exists():raise FileExistsError('Preserve accepted new calculation, do not repeat')
    prior=OUT/'Lab8_Rental4_energy_LightGen_20261029_01/combined_summary.json'
    receipt=prior.parent/'lab_delivery_receipt01.json'
    if sha(receipt)!='2eacfecedc4a6988dbe8d88631ed3c08ba97ab1d6790ba3bddac79c35abb3cbf':raise ValueError('Accepted table delivery pin')
    pins=read(receipt)['files'];record=next(x for x in pins if x['path'].endswith('/combined_summary.json'))
    if sha(prior)!=record['sha256']:raise ValueError('Accepted all36 source changed')
    all36=read(prior);lgfile=OUT/'LightGenII_resize_normalization_20261029_01/LightGenII_timing_summary.json'
    if sha(lgfile)!='08b4682698b2f67f16435444d35737677d61728c1ada5cc7a564eb9fcfb227ab':raise ValueError('Lab LG source pin')
    lab=read(lgfile);lab_rounds=[];files=[]
    for i,pin in enumerate(lab['rounds'],1):
        leaf=SIM/'lgvq_compute_scope_20261029_05'/f'lg_repeat{i:02d}'
        report=leaf/'compute_report.json';audit=leaf/'local_independent_compute_audit01.json';rawfile=leaf/'raw_operator_records.json'
        if sha(report)!=pin['report_sha256'] or sha(audit)!=pin['audit_sha256'] or not pin['all_scores_bitwise_equal']:raise ValueError('Exact lab triple required')
        r=read(report)
        if sha(rawfile)!=r['raw_operator_records_sha256']:raise ValueError('Original lab raw pin')
        lab_rounds.append(entry(r,read(rawfile),slm_hz=slm_hz))
        files.extend(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)) for p in [report,audit,rawfile])
    root=SIM/'lightgen_rental_20261029_01';leaf=root/'lg_repeat01'
    receipt=root/'whole_root_backup_receipt_complete01.json'
    if sha(receipt)!='0de885310273889e6f90f49f870004a4178b7f034ab0eaa7380e0d2755d843a9':raise ValueError('Trusted original rental receipt')
    manifest={r['name']:r['sha256'] for r in read(receipt)['files']}
    for name in ['raw_kernel_trace.json','raw_operator_records.json','numeric_audit.json']:
        p=leaf/name
        if sha(p)!=manifest['lg_repeat01/'+name]:raise ValueError('Original rental raw pin')
        files.append(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)))
    # Preserve the numerical failure. This requested new parallel calculation is
    # diagnostic, not another original-run audit or a forged compute_report.
    raw=read(leaf/'raw_operator_records.json');numeric=read(leaf/'numeric_audit.json')
    if numeric['all_formal_scores_bitwise_equal'] or numeric['equal_scores']!=92:raise ValueError('Actual numeric qualification retained')
    kernels=scope.traced_kernels(read(leaf/'raw_kernel_trace.json'))
    r=prep.summarize_compute(kernels,raw['patch'],raw['tokenizer'],32)
    rental=entry(r,raw,rental=True,slm_hz=slm_hz)
    files.extend(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)) for p in [prior,lgfile])
    def pool(rows):
        fields=sum(x['fields'] for x in rows);seconds=sum(x['total_s'] for x in rows)
        stages=set().union(*(x['equivalent_s_per_group'] for x in rows));work=set().union(*(x['work_s_per_group'] for x in rows))
        return dict(fields=fields,rounds=len(rows),time16_s=seconds/fields,total_s=seconds,native_tokens=sum(x['native_tokens'] for x in rows),
            equivalent_s_per_group={k:sum(x['equivalent_s_per_group'].get(k,0)*x['fields'] for x in rows)/fields for k in stages},
            work_s_per_group={k:sum(x['work_s_per_group'].get(k,0)*x['fields'] for x in rows)/fields for k in work},
            component_energy16_j=sum(x['component_energy']['energy_j'] for x in rows)/fields,
            component_energy=rows[0]['component_energy'],source_rounds=rows)
    lab_result=pool(lab_rounds);rental_result=pool([rental])
    if not standard1886:close(lab_result['time16_s'],lab['LG_time16_s'])
    value=dict(status='LGVQ_ONLY_LG_FIRST_SAVED_DATA_REORGANIZATION',baselines=all36['records'],trials=all36['trials'],
        LG_lab=lab_result,LG_rental=rental_result,LG_rental_software=all36['LightGen_rental'],
        common_tokens16=lab['Qwen_effective_tokens']/96,component_powers_w=lab_result['component_energy']['powers_w'],
        physical_optical_time_measured=False,physical_hybrid_energy_measured=False,rental_numeric_gate_passed=False,
        token_basis='common Qwen input numerator; native LG shared prompt mask counted once per16-video field',
        calculation='original two-field resource schedule, same measured CPU worker assignments and fixed optical cycles; not serial sum or wall/16',
        source_files=files,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),torch_imported='torch' in sys.modules)
    if standard1886:
        frozen=DEST/'lgvq_reorganized_sources01.json'
        if sha(frozen)!='9873a8a7c021a977dc67682b1dcb618fbbaabf7f4944eb4ed2e137bcc1decc1d':raise ValueError('Original frozen population pin')
        old=read(frozen)
        if value['baselines']!=old['baselines'] or value['trials']!=old['trials']:raise ValueError('All36 baseline observations must remain unchanged')
        hardware=dest/'hardware_inventory01.json'
        value.update(status='LGVQ_STANDARD_RESIZE_PATCH1886_ONLY',SLM_hz=1886,optical_layer_s=1/1886+.000300+.0000307,
            includes_full_frontend_preparation=False,CPU_dispatch_and_queues_excluded=True,hardware=read(hardware),
            hardware_inventory_sha256=sha(hardware),old_standard1400=dict(lab_ms16=old['LG_lab']['time16_s']*1000,rental_ms16=old['LG_rental']['time16_s']*1000))
        value['source_files'] += [dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)) for p in [frozen,hardware]]
        close(value['LG_lab']['time16_s']*1000,45.811069885871085)
        close(value['LG_rental']['time16_s']*1000,29.194761633918024)
    if value['torch_imported']:raise RuntimeError('CPU-only collection imported Torch')
    dest.mkdir(parents=True,exist_ok=True)
    with file.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=True,indent=2,allow_nan=False)
    return dict(path=str(file),sha256=sha(file),lab_ms=lab_result['time16_s']*1000,rental_parallel_ms=rental_result['time16_s']*1000,
                lab_proxy_J16=lab_result['component_energy16_j'],rental_proxy_J16=rental_result['component_energy16_j'],torch_imported=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--standard1886',action='store_true');args=parser.parse_args()
    print(json.dumps(collect(args.standard1886),ensure_ascii=True))
