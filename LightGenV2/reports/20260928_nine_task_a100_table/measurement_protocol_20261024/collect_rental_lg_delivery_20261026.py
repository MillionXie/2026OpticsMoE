"""Assemble only sealed raw evidence; never import a model or contact a host."""
import argparse,csv,hashlib,json,math
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

def assemble(out):
    assert not out.exists(),'Preserve prior delivery; use a new immutable folder'
    roots=[RUN/'a100_corrected_boundaries_20261026_02',RUN/'a100_corrected_boundaries_20261026_03',
           RUN/'a100_lowlevel_capture_20261026_01',RUN/'a100_scoreonly_20261026_01',RUN/'a10080_corrected_serial_20261026_01']
    sealed={}
    for root in roots:
        assert read(root/'suite_status.json')['status']=='complete',str(root)
        for receipt in root.rglob('backup_receipt_complete01.json'):
            for rec in read(receipt)['verified_files']:
                p=receipt.parent/rec['path'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],str(p)
                sealed[str(p.resolve())]=rec
    prior=read(BASE/'corrected_scope_20261026_03/independent_results.json');assert prior['status']=='passed'
    qroot=roots[0]/'qwen01';q=read(qroot/'timing.json')
    rows=q['records'];full=(rows[-1]['GPU_ready_host']-rows[0]['CPU_start'])*1000
    removed=sum((r['H2D_end']-r['CPU_start'])*1000-union_ms(r['tokenizer_intervals']) for r in rows)
    close(full-removed,q['corrected_GPU_ready_ms']);close(q['time16_ms'],q['corrected_GPU_ready_ms']/128)
    models=[dict(key='Qwen50',label='Qwen MPS50%',platform='40GB PCIe / 250W',N=2048,
                 full_ms=full,image_H2D_ms=removed,bookkeeping_ms=0.,excluded_GPU_ms=0.,excluded_capture_ms=0.,fixed_ms=0.,
                 total_ms=full-removed,time16_ms=q['time16_ms'],frame_ms=q['equivalent_frame_ms'],
                 tokenizer_ms=q['tokenizer_CPU_ms'],gpu_model_diagnostic_ms=q['GPU_model_window_sum_ms'],source=str((qroot/'timing.json').resolve()),
                 note='复用本口径已合格单轮。严格逐视频串行，无预取、无GPU指令图。')]
    serial=[]
    for model,root,t in [('Qwen50',qroot,q)]+[(k,roots[-1]/(k+'50_repeat01'),read(roots[-1]/(k+'50_repeat01/timing.json'))) for k in ('kimi','deepseek')]:
        if model!='Qwen50':
            audit=read(root/'local_independent_audit01.json');assert audit['status']=='passed'
            assert audit['source_sha256']==sha(root/'timing.json') and audit['four_cycles_predictions_bitwise_equal']
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
    paths=set()
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.csv','.npz','.log','.txt','.yaml','.yml') and 'mps_logs' not in p.parts:paths.add(p.resolve())
    reference=RUN/'a100_lgvq_graph_pipeline_20261026_03/lg_repeat01/graph/graph_outputs.npz';paths.add(reference.resolve())
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),verification='原远端封存SHA/size重核' if str(p) in sealed else '本地审计/收据/外日志或引用本体的本轮SHA') for p in sorted(paths)]
    result=dict(status='passed',models=models,serial_rows=serial,LG_rows=lgrows,LG_diagnostics=diagnostics,source_files=files,
                N=2048,unique_videos=512,frames_per_video=4,videos_per_field=16,fields=128,
                Qwen100_threshold_time16_ms=q['time16_ms']/100,physical_fixed_time16_ms=fixed/128,
                original_512_bytes_shared=True,new_energy_measured=False,all_rounds=1)
    out.mkdir(parents=True)
    (out/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'raw_file_index.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256','verification']);writer.writeheader();writer.writerows(files)
    print(json.dumps(dict(status='passed',output=str(out),files=len(files),remote_sealed_files=len(sealed),serial_records=len(serial),LG_fields=len(lgrows),models=[dict(model=m['key'],time16_ms=m['time16_ms']) for m in models]),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'rental_lg_followup_20261026_01');assemble(p.parse_args().out.resolve())
