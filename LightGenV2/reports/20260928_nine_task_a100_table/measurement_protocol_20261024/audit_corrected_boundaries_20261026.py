"""Independent data-only arithmetic, serial-scope, output and receipt audit."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6),(a,b)
def audit(output):
    qroot=RUN/'a100_corrected_boundaries_20261026_02';lroot=RUN/'a100_corrected_boundaries_20261026_03'
    q=read(qroot/'qwen01/timing.json');lg=read(lroot/'lg01/timing.json')
    assert q['video_calls']==lg['video_calls']==2048 and q['unique_videos']==lg['unique_videos']==512
    for root in (qroot,lroot):assert read(root/'suite_status.json')['status']=='complete'
    assert len(q['records'])==2048 and len(lg['records'])==128
    qrows=q['records'];token=sum((t['end']-t['start'])*1000 for r in qrows for t in r['tokenizer_intervals'])
    removed=sum((r['H2D_end']-r['CPU_start'])*1000 for r in qrows)-token
    close(removed,q['removed_image_preparation_and_H2D_ms']);close(token,q['tokenizer_CPU_ms'])
    qms=q['full_before_subtraction_ms']-removed;close(qms,q['corrected_GPU_ready_ms'])
    close(sum(r['GPU_end_ms']-r['GPU_start_ms'] for r in qrows),q['GPU_model_window_sum_ms'])
    for i,r in enumerate(qrows):
        assert r['index']==i and r['source_index']==i%512 and r['cycle']==i//512
        assert r['CPU_start']<=r['CPU_end']<=r['H2D_end']<=r['GPU_host_start']<=r['GPU_ready_host']<=r['CPU_result_return']
        if i:assert r['CPU_start']>=qrows[i-1]['CPU_result_return']
    kinds=('optical_simulation','detector_ROI','electronic_residual');expected=(6,6,4)
    bykind={k:0. for k in kinds};capkind={k:0. for k in kinds}
    assert len(lg['capture']['segments'])==33
    for s in lg['capture']['segments']:
        if s['kind'] in capkind:capkind[s['kind']]+=s['capture_host_ms']
    for i,r in enumerate(lg['records']):
        assert r['field']==i
        assert r['field_start']<=r['tokenizer_start']<=r['tokenizer_end']<=r['H2D_start']<=r['H2D_end']<=r['score_ready']<=r['bookkeeping_start']<=r['bookkeeping_end']
        if i:assert r['field_start']>=lg['records'][i-1]['score_ready']
        for k,n in zip(kinds,expected):
            spans=[s for s in r['optical_simulation_spans'] if s['kind']==k];assert len(spans)==n
            assert all(s['cuda_ms']>=0 for s in spans);bykind[k]+=sum(s['cuda_ms'] for s in spans)
    h2d=sum((r['H2D_end']-r['H2D_start'])*1000 for r in lg['records'])
    book=sum((r['bookkeeping_end']-r['bookkeeping_start'])*1000 for r in lg['records'][:-1])
    for k in kinds:close(bykind[k],lg['excluded_GPU_ms_by_kind'][k]);close(capkind[k],lg['excluded_capture_CPU_ms_by_kind'][k])
    close(h2d,lg['removed_original_H2D_ms']);close(book,lg['removed_measurement_only_bookkeeping_ms'])
    charged=lg['full_software_host_ms']-h2d-book-sum(bykind.values())-sum(capkind.values())
    close(charged,lg['charged_actual_electronic_ms']);assert charged>0
    optical=128*6*(1000/1400+.300+.0307);close(optical,lg['fixed_optical_ms']);lms=charged+optical;close(lms,lg['projection_ms'])
    for root,leaf,cap in ((qroot,'qwen01',50),(lroot,'lg01',100)):
        identity=read(root/leaf/'identity.json');assert identity['cap']==cap
        guard=read(root/leaf/'process_audit.json');assert guard
        for row in guard:
            assert not row['foreign_gpu_processes'] and not row['foreign_mps_clients']
            assert row['platform']['requested_cap']==cap and row['platform']['power_limit_w']==250
            assert row['platform']['uuid']=='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
        for rel,digest in identity['sources'].items():
            blob=subprocess.check_output(['git','show',identity['published_commit']+':'+rel],cwd=REPO)
            assert hashlib.sha256(blob).hexdigest()==digest,rel
    projection=read(lroot/'lg01/projection_identity.json')
    for rel,digest in projection['sources'].items():
        blob=subprocess.check_output(['git','show',projection['module_deployment']['published_commit']+':'+rel],cwd=REPO)
        assert hashlib.sha256(blob).hexdigest()==digest,rel
    qfiles=read(qroot/'qwen01/input_files.json');lfiles=read(lroot/'lg01/input_files.json');assert qfiles==lfiles and len(qfiles)==512
    with np.load(qroot/'qwen01/outputs.npz',allow_pickle=False) as f:qa=f['prediction'].copy()
    with np.load(lroot/'lg01/outputs.npz',allow_pickle=False) as f:la=f['prediction'].copy()
    assert qa.shape==(2048,) and la.shape==(128,16)
    for c in range(4):assert np.array_equal(qa[:512],qa[c*512:(c+1)*512]) and np.array_equal(la[:32],la[c*32:(c+1)*32])
    ref=lg['source_reference_for_validation_only'];rpath=REPO/Path(ref['path']).relative_to('/DATA/DATA1/guest3/2026OpticsMoE')
    assert sha(rpath)==ref['sha256']
    with np.load(rpath,allow_pickle=False) as f:assert np.array_equal(la,f['prediction'][:128])
    indexed=[];receipt_verified=0
    for root in (qroot,lroot):
        receipt=read(root/'backup_receipt_complete01.json')
        entries=receipt['verified_files']
        for e in entries:
            rel=e.get('relative_path',e.get('path'));p=root/rel
            assert p.stat().st_size==e.get('bytes',e.get('size')) and sha(p)==e['sha256'],str(p)
            receipt_verified+=1
        for p in sorted(root.rglob('*')):
            if p.is_file():indexed.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
    with (output/'raw_file_index.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(indexed)
    result=dict(status='passed',N=2048,unique_videos=512,fields=128,frames_per_field=64,rounds_per_model=1,
        Qwen=q,LG=lg,Qwen_total_ms=qms,LG_total_ms=lms,Qwen_time16_ms=qms/128,LG_time16_ms=lms/128,speedup=qms/lms,
        Qwen_frame_ms=qms/8192,LG_frame_ms=lms/8192,excluded_GPU_ms_by_kind=bykind,excluded_capture_CPU_ms_by_kind=capkind,
        retained_Qwen_root=str(qroot),LG_root=str(lroot),source_files=indexed,receipt_verified_files=receipt_verified,
        same_512_input_paths_size_SHA=True,all_original_predictions_bitwise_equal=True,exclusive_process_audits_passed=True,
        observed_scope='actual retained electronics plus fixed optics; ROI and electronic residual execute but timing excluded')
    (output/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('Qwen','LG','source_files')},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);audit(a.output)
