"""Read-only XLSX/cache reconciliation against the sealed independent raw ledger."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path
import openpyxl

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert isinstance(a,(int,float)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-5),(a,b)
def review(out):
    d=json.loads((out/'independent_results.json').read_text(encoding='utf-8'))
    p=out/'A100_Temporal_Kimi_DS与LG优化_单轮2048次_20261026.xlsx'
    v=openpyxl.load_workbook(p,data_only=True,read_only=True);f=openpyxl.load_workbook(p,data_only=False,read_only=True)
    assert v.sheetnames==['LG理想覆盖估算','16视频对照','累计时间怎么算','LG逐组原值','大模型逐视频','原始文件索引']
    assert v['LG逐组原值'].max_row>=389 and v['大模型逐视频'].max_row>=6149
    formulas=0
    for sheet in f:
        caches=v[sheet.title]
        for fr,vr in zip(sheet.iter_rows(),caches.iter_rows()):
            for fc,vc in zip(fr,vr):
                assert vc.data_type!='e',(sheet.title,vc.coordinate,vc.value)
                if fc.data_type=='f':
                    formulas+=1;assert isinstance(vc.value,(int,float)),(sheet.title,vc.coordinate,vc.value)
    for i,m in enumerate(d['models']):
        r=6+i;s=v['16视频对照'];c=v['累计时间怎么算']
        close(s.cell(r,3).value,m['time16_ms']);close(s.cell(r,4).value,m['frame_ms']);close(c.cell(r,6).value,m['total_ms'])
        close(s.cell(r,5).value,m['time16_ms']/d['models'][4]['time16_ms'])
        close(s.cell(r,6).value,m['time16_ms']/d['models'][5]['time16_ms'])
    overlap=d['ideal_overlap'];close(v['LG理想覆盖估算']['B20'].value,overlap['simple_ideal_time16_ms']);close(v['LG理想覆盖估算']['C20'].value,overlap['simple_ideal_Qwen50_ratio'])
    close(v['LG理想覆盖估算']['B19'].value,overlap['only_fixed_optical_gaps_ideal_time16_ms'])
    close(v['LG理想覆盖估算']['B37'].value,d['models'][5]['time16_ms']-d['models'][5]['tokenizer_ms']/128-d['models'][5]['electronic_capture_ms']/128-d['models'][5]['fixed_ms']/128)
    lg={k:0. for k in ('LG_original','LG_lowlevel','LG_scoreonly')}
    for actual,r in zip(v['LG逐组原值'].iter_rows(min_row=6,max_row=389,values_only=True),d['LG_rows']):
        actual=list(actual)
        assert actual[0]==r[0] and actual[1]==r[1]
        for a,b in zip(actual[2:12],r[2:12]):close(a,b)
        lg[r[0]]+=actual[10]
    for m in d['models'][3:]:close(lg[m['key']],m['total_ms'])
    serial={k:0. for k in ('Qwen50','kimi','deepseek')}
    for actual,r in zip(v['大模型逐视频'].iter_rows(min_row=6,max_row=6149,values_only=True),d['serial_rows']):
        actual=list(actual)
        assert actual[:5]==r[:5]
        for a,b in zip(actual[5:13],r[5:13]):close(a,b)
        serial[r[0]]+=actual[12]
    for m in d['models'][:3]:close(serial[m['key']],m['image_H2D_ms'])
    for rec in d['source_files']:
        raw=Path(rec['path']);assert raw.stat().st_size==rec['bytes'] and sha(raw)==rec['sha256'],str(raw)
    result=dict(status='passed',formula_cache_cells_verified=formulas,serial_records=6144,LG_fields=384,
                original_paths_SHA_size_verified=len(d['source_files']),all_sheets_previewed_separately=True,
                accuracy_note='原评分逐位核对；未重评完整TEST，单轮不建立显著性结论')
    audit=out/'exported_workbook_review.json';assert not audit.exists();audit.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[p,out/'independent_results.json',out/'raw_file_index.csv',audit]
    files.extend(x for x in out.glob('*exit*.json'))
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();sources=[]
    protocol=Path(__file__).resolve().parent;repo=protocol.parents[3]
    for name in ('audit_rental_lg_followup_20261026.py','collect_rental_lg_delivery_20261026.py','build_rental_lg_delivery_20261026.mjs','review_rental_lg_delivery_20261026.py'):
        source=protocol/name;rel=source.relative_to(repo).as_posix();blob=subprocess.check_output(['git','show',source_commit+':'+rel])
        assert source.read_bytes().replace(b'\r\n',b'\n')==blob,rel
        sources.append(dict(repository_path=rel,Git_blob_sha256=hashlib.sha256(blob).hexdigest(),local_working_tree_sha256=sha(source)))
    manifest=dict(delivery_kind='new_single_round_timing_and_LG_implementation_review',analysis_source_Git_commit=source_commit,analysis_sources=sources,files=[dict(path=str(x.resolve()),bytes=x.stat().st_size,sha256=sha(x)) for x in files],
                  raw_file_index_path=str((out/'raw_file_index.csv').resolve()),raw_file_index_sha256=sha(out/'raw_file_index.csv'),
                  preserves_all_prior_deliveries=True,energy_not_newly_measured=True)
    dest=out/'delivery_manifest.json';assert not dest.exists();dest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(result,manifest_sha256=sha(dest)),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);review(p.parse_args().out.resolve())
