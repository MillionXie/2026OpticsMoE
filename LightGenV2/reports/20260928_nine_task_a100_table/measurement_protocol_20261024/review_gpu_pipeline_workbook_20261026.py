"""Independent read-only XLSX cache/raw-index checks, then delivery manifest."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import openpyxl

HERE=Path(__file__).resolve().parent


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def near(a,b):assert isinstance(a,(int,float)) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5),(a,b)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path)
    cli.add_argument('--visual-reviewed',action='store_true');args=cli.parse_args();root=args.output
    assert args.visual_reviewed,'Inspect all four rendered worksheets before sealing delivery'
    assert not (root/'delivery_manifest.json').exists(),'Preserve sealed delivery'
    report=json.loads((root/'independent_results.json').read_text(encoding='utf-8'));assert report['status']=='passed'
    file=root/'A100_LG流水线与Qwen50严格串行_2048次对照_20261026.xlsx'
    cached=openpyxl.load_workbook(file,data_only=True,read_only=True)
    formula=openpyxl.load_workbook(file,data_only=False,read_only=True)
    nformulas=0
    for sh in formula:
        values=cached[sh.title]
        for row in sh:
            for c in row:
                v=values[c.coordinate]
                assert v.data_type!='e',(sh.title,c.coordinate,v.value)
                if c.data_type=='f':
                    nformulas+=1;assert v.value is not None,(sh.title,c.coordinate,'missing formula cache')
    for cell in ('B6','C6','D6','E6','F6','D7','D8','E15','D34'):
        assert formula['对照结果'][cell].data_type=='f',cell
    for row,p in zip((6,7,8),report['aggregates']):
        sh=cached['对照结果'];near(sh[f'B{row}'].value,p['total_video_calls'])
        near(sh[f'C{row}'].value,p['total_elapsed_ms']/1000)
        near(sh[f'D{row}'].value,p['time16_ms']);near(sh[f'E{row}'].value,p['equivalent_frame_ms'])
        near(sh[f'F{row}'].value,p['Qwen50_over_system_ratio'])
    for row,n in zip((15,16,17,18),(256,512,1024,2048)):
        for col,system,mode in [('B','LG','serial'),('C','LG','pipeline'),('D','Qwen','native_serial')]:
            points=[p for p in report['all_points'] if p['system']==system and p['mode']==mode]
            expected=sum(next(x['time16_ms'] for x in p['prefixes'] if x['video_calls']==n) for p in points)/3
            near(cached['对照结果'][f'{col}{row}'].value,expected)
    for i,p in enumerate(report['extra_preparation']):
        count=p['actual_CPU_prepared_video_calls']
        near(cached['节点与分项'][f'D{30+i}'].value,p['recorded_CPU_tokenizer_ms']*16/count)
        cpu=p['CPU_preparation_span_ms'] if p['system']=='lg' else p['CPU_prepare_total_ms']
        transfer=p['packing_transfer_tokenizer_window_sum_ms'] if p['system']=='lg' else p['H2D_total_ms']
        near(cached['时间花在哪里'][f'C{15+i}'].value,cpu*16/count)
        near(cached['时间花在哪里'][f'D{15+i}'].value,transfer*16/count)
    cached.close();formula.close()
    with (root/'raw_file_index.csv').open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream))
    for r in records:
        p=Path(r['path']);assert p.stat().st_size==int(r['bytes']) and digest(p)==r['sha256']
    review=dict(status='passed',formula_count=nformulas,formula_caches_nonempty_and_error_free=True,
        independently_recomputed_summary_and_prefix_values=True,verified_raw_files=len(records),
        verified_raw_bytes=sum(int(r['bytes']) for r in records),all_four_sheet_renders_visually_reviewed=True,
        no_models_loaded_or_run=True)
    (root/'independent_workbook_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[file,*[root/name for name in ('independent_results.json','raw_file_index.csv','workbook_formula_review.json',
        'independent_workbook_review.json','summary.png','raw.png','nodes.png','parts.png')]]
    sources=[HERE/name for name in ('lgvq_gpu_pipeline_20261026.py','lgvq_minimal_pipeline_20261026.py','qwen_strict_serial_20261026.py','qwen_shared_serial_20261026.py',
        'audit_gpu_pipeline_20261026.py','build_gpu_pipeline_20261026.mjs','review_gpu_pipeline_workbook_20261026.py')]
    entry=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p))
    manifest=dict(status='passed',profile='LG_optimized_GPU_pipeline_vs_Qwen_strict_interleaved_single_video_20261026',
        Qwen_nonexclusive_authorized=report['Qwen_nonexclusive_authorized'],
        ratios_eligible_for_uncontended_hardware_speedup_claim=report['ratios_eligible_for_uncontended_hardware_speedup_claim'],
        deliverables=[entry(p) for p in files],generator_and_reviewer_sources=[entry(p) for p in sources],
        scope='Same512 original inputs repeated4times per mode,3fresh processes. LG raw-GPU-input optimization; Qwen prepares/transfers/infers/returns each video before starting next. Not symmetric file-input E2E or physical optical measurement.',
        raw_index_contains_exact_paths_sizes_SHA256=True,old_artifacts_preserved=True)
    target=root/'delivery_manifest.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(review=review,manifest=entry(target)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
