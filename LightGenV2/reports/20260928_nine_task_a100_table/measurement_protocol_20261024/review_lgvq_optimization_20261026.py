"""Read exported XLSX caches independently; write exact raw index and manifest."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import openpyxl

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def info(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def near(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7),(a,b)
def write_new(path,value):
    content=json.dumps(value,indent=2,ensure_ascii=False).encode('utf-8')
    if path.exists():assert path.read_bytes()==content,'Immutable output differs'
    else:path.write_bytes(content)

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--visual-reviewed',action='store_true',required=True);a=p.parse_args();out=a.output
    result=read(out/'independent_results.json');assert result['status']=='passed'
    book=next(out.glob('*.xlsx'));formulas=openpyxl.load_workbook(book,data_only=False);cached=openpyxl.load_workbook(book,data_only=True)
    count=0
    for sheet in formulas:
        for row in sheet:
            for cell in row:
                value=cached[sheet.title][cell.coordinate].value
                assert cached[sheet.title][cell.coordinate].data_type!='e',(sheet.title,cell.coordinate,value)
                if cell.data_type=='f':
                    count+=1;assert isinstance(value,(int,float)) and math.isfinite(value),(sheet.title,cell.coordinate,value)
    summary=cached['16视频结果与口径'];raw=cached['逐轮原值与前缀']
    points=result['all_points'];kinds=list(dict.fromkeys(r['kind'] for r in points));totals={}
    for kind in kinds:
        group=[r for r in points if r['kind']==kind];assert len(group)==3
        totals[kind]=sum(r['optimized_GPU_total_ms'] for r in group)/384
    selected=min(totals,key=totals.get);lg=totals[selected]
    q=sum(r['elapsed_ms'] for r in result['Qwen_retained'])/384
    qgpu=sum(r['GPU_model_window_sum16_ms'] for r in result['Qwen_retained'])/3
    near(summary['B6'].value,q);near(summary['B7'].value,qgpu)
    for r in (6,7):near(summary[f'C{r}'].value,lg)
    near(summary['D6'].value,q/lg);near(summary['D7'].value,qgpu/lg)
    for i,point in enumerate(points):
        r=5+i;near(raw[f'F{r}'].value,point['control_GPU_total_ms']/128);near(raw[f'G{r}'].value,point['optimized_GPU_total_ms']/128);near(raw[f'H{r}'].value,point['optimized_GPU_total_ms']/8192)
    report=dict(status='passed',formula_cells_checked=count,formula_errors=0,selected_by_all_three_round_mean=selected,
                LG_ms_per16=lg,LG_ms_per_frame=lg/64,Qwen_full_ms_per16=q,Qwen_GPU_model_sum_ms_per16=qgpu,
                requested_operational_reference_ratio=q/lg,GPU_model_window_reference_ratio=qgpu/lg,
                paired_LG_speedup={k:sum(r['control_GPU_total_ms'] for r in points if r['kind']==k)/sum(r['optimized_GPU_total_ms'] for r in points if r['kind']==k) for k in kinds},
                original_interval_and_all_output_review='independent_results.json',
                human_visual_review=['results.png','explanation.png','rounds.png'],
                uncontended_or_symmetric_file_E2E_100x_claim_supported=False)
    write_new(out/'independent_workbook_review.json',report)
    # Remote names only come from actual transfer receipts, never inferred for
    # local CPU audits or receipt files. External log origin is separately known.
    remote={}
    root=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
    for experiment in root.glob('a100_*20261026_*'):
        for receipt_path in experiment.glob('**/backup_receipt_*.json'):
            receipt=read(receipt_path)
            for r in receipt.get('verified_files',[]):remote[str((receipt_path.parent/r['path']).resolve())]=receipt['source'].rstrip('/')+'/'+r['path']
        logfile=experiment/'external_controller.log'
        if logfile.exists():remote[str(logfile.resolve())]='/DATA/DATA1/guest3/2026OpticsMoE/'+experiment.relative_to(REPO).as_posix()+'.controller.log'
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['local_exact_path','remote_path_if_receipted','bytes','sha256']);writer.writeheader()
    for row in sorted(result['source_files'],key=lambda r:r['path']):
        f=Path(row['path']);assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
        writer.writerow(dict(local_exact_path=str(f.resolve()),remote_path_if_receipted=remote.get(str(f.resolve()),''),bytes=row['bytes'],sha256=row['sha256']))
    index=out/'raw_file_index.csv';content=stream.getvalue().encode('utf-8-sig')
    if index.exists():assert index.read_bytes()==content
    else:index.write_bytes(content)
    deliver=[book,out/'independent_results.json',index,out/'formula_review.json',out/'independent_workbook_review.json']
    manifest=dict(status='passed',profile='LG original arithmetic graph and pipeline optimization, Qwen retained strict serial',
                  actual_software_GPU_measurement=True,physical_optical_E2E_measurement=False,
                  no_fair_symmetric_file_E2E_100x_claim=True,raw_file_count=len(result['source_files']),
                  all_rounds_retained=True,deliverables=[info(f) for f in deliver],visual_reviewed=True,
                  review_summary=report)
    write_new(out/'delivery_manifest.json',manifest)
    print(json.dumps(dict(workbook=info(book),manifest=info(out/'delivery_manifest.json'),raw_index=info(index),review=report),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
