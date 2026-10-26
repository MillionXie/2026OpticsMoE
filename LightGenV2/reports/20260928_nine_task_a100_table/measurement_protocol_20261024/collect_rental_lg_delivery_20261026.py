"""Assemble only sealed raw evidence; never import a model or contact a host."""
import argparse,ast,csv,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
PROTOCOL=Path(__file__).resolve().parent
BASE=PROTOCOL/'outputs/01a061e2-6f88-7913-87f5-b632000e4c83'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-5),(a,b)
def union_ms(intervals):
    total=0.;end=-math.inf
    for t in sorted(intervals,key=lambda x:x['start']):
        total+=max(0,t['end']-max(end,t['start']));end=max(end,t['end'])
    return total*1000

def assemble(out,rebuild_own=False):
    assert not out.exists() or (rebuild_own and not (out/'delivery_manifest.json').exists()),'Preserve sealed delivery; use a new immutable folder'
    roots=[RUN/'a100_corrected_boundaries_20261026_02',RUN/'a100_corrected_boundaries_20261026_03',
           RUN/'a100_lowlevel_capture_20261026_01',RUN/'a100_scoreonly_20261026_01',RUN/'a10080_corrected_serial_20261026_01']
    sealed={}
    for root in roots:
        suite=read(root/'suite_status.json')
        if root==roots[-1]:
            assert suite['status']=='complete' or (suite['status']=='stopped' and suite['mps_released'] and read(root/'deepseek50_repeat01/local_independent_audit01.json')['qualification']=='complete_2048_inference_qualified_by_postwrite_data_only_seal')
        else:assert suite['status']=='complete',str(root)
        for receipt in root.rglob('backup_receipt_complete01.json'):
            for rec in read(receipt)['verified_files']:
                p=receipt.parent/rec['path'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],str(p)
                sealed[str(p.resolve())]=rec
    prior=read(BASE/'corrected_scope_20261026_03/independent_results.json');assert prior['status']=='passed'
    qroot=roots[0]/'qwen01';q=read(qroot/'timing.json')
    rows=q['records'];record_span=(rows[-1]['GPU_ready_host']-rows[0]['CPU_start'])*1000
    # The retained Qwen entry starts full_start immediately before its loop,
    # 0.064767ms before first CPU_start. Preserve this original dispatch lead-in.
    leadin=q['full_before_subtraction_ms']-record_span;assert 0<=leadin<1
    full=record_span+leadin
    removed=sum((r['H2D_end']-r['CPU_start'])*1000-union_ms(r['tokenizer_intervals']) for r in rows)
    close(full-removed,q['corrected_GPU_ready_ms']);close(q['time16_ms'],q['corrected_GPU_ready_ms']/128)
    models=[dict(key='Qwen50',label='Qwen MPS50%',platform='40GB PCIe / 250W',N=2048,
                 full_ms=full,image_H2D_ms=removed,bookkeeping_ms=0.,excluded_GPU_ms=0.,excluded_capture_ms=0.,fixed_ms=0.,
                 total_ms=full-removed,time16_ms=q['time16_ms'],frame_ms=q['equivalent_frame_ms'],retained_entry_dispatch_ms=leadin,
                 tokenizer_ms=q['tokenizer_CPU_ms'],gpu_model_diagnostic_ms=q['GPU_model_window_sum_ms'],source=str((qroot/'timing.json').resolve()),
                 note='复用本口径已合格单轮。严格逐视频串行，无预取、无GPU指令图。')]
    serial=[]
    for model,root,t in [('Qwen50',qroot,q)]+[(k,roots[-1]/(k+'50_repeat01'),read(roots[-1]/(k+'50_repeat01/timing.json'))) for k in ('kimi','deepseek')]:
        if model!='Qwen50':
            audit=read(root/('local_independent_audit02.json' if (root/'local_independent_audit02.json').exists() else 'local_independent_audit01.json'));assert audit['status']=='passed'
            assert audit['source_sha256']==sha(root/'timing.json') and audit['four_cycles_predictions_bitwise_equal']
            assert audit['same_original_frame_positions_as_Qwen']
            models.append(dict(key=model,label='Kimi MPS50%' if model=='kimi' else 'DS完整模型 MPS50%',platform='80GB SXM4 / 400W',N=2048,
                 full_ms=audit['full_before_subtraction_ms'],image_H2D_ms=audit['removed_image_preparation_and_H2D_ms'],bookkeeping_ms=0.,excluded_GPU_ms=0.,excluded_capture_ms=0.,fixed_ms=0.,
                 total_ms=audit['corrected_total_ms'],time16_ms=audit['time16_ms'],frame_ms=audit['equivalent_frame_ms'],tokenizer_ms=audit['tokenizer_CPU_ms'],gpu_model_diagnostic_ms=audit['GPU_model_cuda_sum_ms'],
                 source=str((root/'timing.json').resolve()),note='本次新租机单轮。与LG/Qwen不是同一张卡，仅作跨配置耗时比。'))
        origin=t['records'][0]['CPU_start']
        for r in t['records']:
            tok=union_ms(r['tokenizer_intervals'])
            serial.append([model,r['index'],r['source_index'],r['cycle'],r['sample_id'],
                           *[(r[k]-origin)*1000 for k in ('CPU_start','CPU_end','H2D_end','GPU_host_start','GPU_ready_host','CPU_result_return')],
                           tok,(r['H2D_end']-r['CPU_start'])*1000-tok])
    lgrows=[];diagnostics=[]
    for key,root,auditfile,label in [('LG_original',roots[1],None,'LG原分段图'),('LG_lowlevel',roots[2],'local_independent_audit02.json','LG减少重复建图准备'),('LG_scoreonly',roots[3],'local_independent_audit01.json','LG只省训练辅助统计')]:
        t=read(root/'lg01/timing.json');assert t['status']=='complete'
        if auditfile:
            audit=read(root/auditfile);assert audit['status']=='passed' and audit['source_sha256']==sha(root/'lg01/timing.json')
            assert audit['all_predictions_bitwise_equal_original']
        else:audit=prior['LG']
        rs=t['records'];assert len(rs)==128
        kinds=('optical_simulation','detector_ROI','electronic_residual')
        excluded={k:sum(x['cuda_ms'] for r in rs for x in r['optical_simulation_spans'] if x['kind']==k) for k in kinds}
        cap=sum(x['capture_host_ms'] for x in t['capture']['segments'] if x['kind'] in kinds)
        h2d=sum((r['H2D_end']-r['H2D_start'])*1000 for r in rs)
        book=sum((r['bookkeeping_end']-r['bookkeeping_start'])*1000 for r in rs[:-1])
        electronic=t['full_software_host_ms']-h2d-book-sum(excluded.values())-cap
        fixed=128*6*(1000/1400+.300+.0307);total=electronic+fixed
        close(total/128,t['time16_ms'])
        models.append(dict(key=key,label=label,platform='40GB PCIe / 250W',N=2048,full_ms=t['full_software_host_ms'],image_H2D_ms=h2d,bookkeeping_ms=book,
                 excluded_GPU_ms=sum(excluded.values()),excluded_capture_ms=cap,fixed_ms=fixed,total_ms=total,time16_ms=total/128,frame_ms=total/8192,
                 tokenizer_ms=sum((r['tokenizer_end']-r['tokenizer_start'])*1000 for r in rs),
                 electronic_capture_ms=sum(x['capture_host_ms'] for x in t['capture']['segments'] if x['kind']=='electronics'),
                 source=str((root/'lg01/timing.json').resolve()),note='每场16视频64帧相干整场，实际计费电子＋固定光周期。'+('辅助字典缩减，仅评分保持逐位一致。' if key=='LG_scoreonly' else '保留完整原评分与辅助计算。')))
        projected=[]
        for i,r in enumerate(rs):
            span=(r['score_ready']-rs[i-1]['score_ready'])*1000 if i else t['full_software_host_ms']-(rs[-1]['score_ready']-r['score_ready'])*1000
            bk=(rs[i-1]['bookkeeping_end']-rs[i-1]['bookkeeping_start'])*1000 if i else 0.
            vals=[sum(x['cuda_ms'] for x in r['optical_simulation_spans'] if x['kind']==k) for k in kinds]
            valsrow=[key,i,span,(r['H2D_end']-r['H2D_start'])*1000,bk,*vals,cap if i==3 else 0.,fixed/128]
            proj=span-sum(valsrow[3:9])+valsrow[9];projected.append(proj);lgrows.append(valsrow+[proj,proj/64])
        close(sum(projected),total)
        diagnostics.append(dict(key=key,post_capture_all124_time16_ms=sum(projected[4:])/124,
                    formal_time16_ms=total/128,description='仅完整后124场诊断，未作为正式分母，未挑最快窗口'))
    # Human-requested quick ideal-overlap estimate. Historical eager stage
    # events are a proxy, NEVER a measurement of the current Graph frontend.
    stage_sources=sorted((RUN/'a100_gpu_pipeline_formal_20261026_01').glob('lg_repeat*/serial_timing.json'))
    assert len(stage_sources)==3
    stage_totals={};stage_count=0
    stage_receipts=[]
    for p in stage_sources:
        old=read(p);assert len(old['records'])==128
        assert read(p.parent/'status.json')['status']=='complete'
        old_identity=read(p.parent/'identity.json');now_identity=read(roots[3]/'lg01/identity.json')
        native_path='LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/lgvq_gpu_pipeline_20261026.py'
        old_blob=subprocess.check_output(['git','show',old_identity['published_measurement_commit']+':'+native_path],cwd=REPO)
        now_blob=subprocess.check_output(['git','show',now_identity['published_commit']+':'+native_path],cwd=REPO)
        assert hashlib.sha256(old_blob).hexdigest()==old_identity['sources'][native_path]
        assert hashlib.sha256(now_blob).hexdigest()==now_identity['sources'][native_path]
        native_AST=lambda blob:ast.dump(next(n for n in ast.parse(blob).body if isinstance(n,ast.FunctionDef) and n.name=='native_lg_front'),include_attributes=False)
        assert native_AST(old_blob)==native_AST(now_blob),'Different frontend mathematics cannot be used even as this proxy'
        receipt_path=p.parent/'backup_receipt_complete01.json'
        if not receipt_path.exists():receipt_path=p.parent.parent/'backup_receipt_stopped01.json'
        stage_receipts.append(receipt_path)
        rr=read(receipt_path)['verified_files']
        for rec in rr:
            member=receipt_path.parent/rec['path'];assert sha(member)==rec['sha256'] and member.stat().st_size==rec['bytes'];sealed[str(member.resolve())]=rec
        for row in old['records'][3:]:
            stage_count+=1
            for stage in row['front_stages']:stage_totals[stage['stage']]=stage_totals.get(stage['stage'],0.)+stage['cuda_event_ms']
    assert stage_count==375
    stages={k:v/stage_count for k,v in stage_totals.items()};frontend_proxy=sum(stages.values())
    current=read(roots[3]/'lg01/timing.json');tok=[(r['tokenizer_end']-r['tokenizer_start'])*1000 for r in current['records']]
    later_CPU=(sum(tok)-tok[0])/128;later_front=frontend_proxy*127/128
    simple=current['time16_ms']-later_CPU-later_front
    overlap=dict(status='THEORETICAL_ONLY_NOT_NEW_OVERLAP_MEASUREMENT',historical_stage_proxy_ms16=stages,historical_frontend_proxy_ms16=frontend_proxy,
        proxy_fields=stage_count,proxy_excludes_first3_each_of3_old_runs=True,proxy_sources=[str(p.resolve()) for p in stage_sources],
        current_CPU_tokenizer_ms16=sum(tok)/128,current_first_CPU_tokenizer_ms=tok[0],CPU_later_covered_ms16=later_CPU,
        GPU_frontend_later_assumed_covered_ms16=later_front,first_field_and_all_electronic_capture_retained=True,
        simple_ideal_time16_ms=simple,simple_ideal_Qwen50_ratio=q['time16_ms']/simple,
        only_fixed_optical_gaps_ideal_time16_ms=current['time16_ms']-later_CPU-min(frontend_proxy,fixed/128)*127/128,
        GPU_same_card_contention_and_fragmented_optical_windows_not_validated=True,
        historical_and_current_native_frontend_function_AST_equal=True,
        current_embedding_GPU_time_not_separately_measured=True)
    paths=set(stage_sources+stage_receipts)
    proof_path=RUN/'a10080_postwrite_data_seal_20261026_01/postwrite_seal.json'
    if proof_path.exists():paths.add(proof_path)
    for p in stage_sources:
        paths.update(p.parent/name for name in ('identity.json','process_audit.json'))
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.csv','.npz','.log','.txt','.yaml','.yml'):paths.add(p.resolve())
    reference=RUN/'a100_lgvq_graph_pipeline_20261026_03/lg_repeat01/graph/graph_outputs.npz';paths.add(reference.resolve())
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),verification='原远端封存SHA/size重核' if str(p) in sealed else '本地审计/收据/外日志或引用本体的本轮SHA') for p in sorted(paths)]
    result=dict(status='passed',models=models,serial_rows=serial,LG_rows=lgrows,LG_diagnostics=diagnostics,source_files=files,ideal_overlap=overlap,
                N=2048,unique_videos=512,frames_per_video=4,videos_per_field=16,fields=128,
                Qwen100_threshold_time16_ms=q['time16_ms']/100,physical_fixed_time16_ms=fixed/128,
                original_512_bytes_shared=True,new_energy_measured=False,all_rounds=1)
    if out.exists():
        previous=read(out/'independent_results.json')
        for name in ('models','serial_rows','LG_rows','LG_diagnostics','ideal_overlap'):assert previous[name]==result[name],name
        actual_files={x['path']:(x['bytes'],x['sha256']) for x in files}
        assert all(actual_files[x['path']]==(x['bytes'],x['sha256']) for x in previous['source_files'])
    out.mkdir(parents=True,exist_ok=rebuild_own)
    (out/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'raw_file_index.csv').open('w' if rebuild_own else 'x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256','verification']);writer.writeheader();writer.writerows(files)
    print(json.dumps(dict(status='passed',output=str(out),files=len(files),remote_sealed_files=len(sealed),serial_records=len(serial),LG_fields=len(lgrows),models=[dict(model=m['key'],time16_ms=m['time16_ms']) for m in models]),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'rental_lg_followup_20261026_01');p.add_argument('--rebuild-own',action='store_true');a=p.parse_args();assemble(a.out.resolve(),a.rebuild_own)
