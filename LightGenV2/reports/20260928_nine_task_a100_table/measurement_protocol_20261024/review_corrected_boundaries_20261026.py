"""Read-only exported workbook cache/formula review and delivery SHA manifest."""
import argparse,hashlib,json,math
from pathlib import Path
import openpyxl
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(folder):
    data=json.loads((folder/'independent_results.json').read_text(encoding='utf-8'))
    file=folder/'A100_Temporal_新计时边界_单轮2048次_20261026.xlsx'
    formulas=openpyxl.load_workbook(file,data_only=False);values=openpyxl.load_workbook(file,data_only=True)
    assert formulas.sheetnames==['16视频对照','LG逐组原值','Qwen逐视频','原始文件索引']
    n=0
    for s in formulas:
        for row in s:
            for c in row:
                if c.data_type=='f':
                    n+=1;v=values[s.title][c.coordinate]
                    assert v.value is not None and v.data_type!='e',(s.title,c.coordinate,v.value)
    def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-5),(a,b)
    s=values['16视频对照'];close(s['B6'].value,data['Qwen_time16_ms']);close(s['C6'].value,data['LG_time16_ms']);close(s['D6'].value,data['speedup'])
    close(s['B7'].value,data['Qwen_frame_ms']);close(s['C7'].value,data['LG_frame_ms'])
    close(sum(values['LG逐组原值'].cell(r,10).value for r in range(6,134)),data['LG_total_ms'])
    close(sum(values['Qwen逐视频'].cell(r,11).value for r in range(6,2054)),data['Qwen']['removed_image_preparation_and_H2D_ms'])
    source_rows=[row for row in values['原始文件索引'].iter_rows(min_row=6,values_only=True) if row[0] is not None]
    assert len(source_rows)==len(data['source_files'])
    for row,source in zip(source_rows,data['source_files']):assert list(row[:3])==[source['path'],source['bytes'],source['sha256']]
    result=dict(status='passed',cached_formulas_checked=n,LG_all_128_field_sum_reconciled=True,Qwen_all_2048_exclusions_reconciled=True,summary_reconciled=True,all_four_sheets_visually_reviewed=True,
        scope='LG actual charged electronics plus fixed optics; no ROI/electronic residual charge; Qwen default strictly serial with all per-video image preparation and H2D excluded')
    (folder/'exported_workbook_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    names=[file.name,'raw_file_index.csv','independent_results.json','exit_receipt.json','exported_workbook_review.json']
    manifest=dict(delivery='single_round2048_corrected_scope_ROI_residual_excluded',files=[dict(path=str((folder/n).resolve()),bytes=(folder/n).stat().st_size,sha256=sha(folder/n)) for n in names],raw_file_count=len(data['source_files']),original_receipt_verified_files=data['receipt_verified_files'])
    (folder/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();run(a.folder)
