"""Independent saved-XLSX checks for the explanatory edition. No model imports."""
import hashlib
import json
import math
import sys
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'outputs/01a061e2-6f88-7913-87f5-b632000e4c83'
OLD = ROOT / 'complete_20261025_01'
NEW = ROOT / 'timing_chinese_notes_20261025_01'
BOOK = NEW / 'A100_LG时间拆分_中文注释与64帧核对_20261025.xlsx'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(a, b):
    assert isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), (a, b)


def main():
    original = json.loads((OLD / 'delivery_manifest.json').read_text(encoding='utf-8'))
    for item in original['outputs']:
        p = OLD / item['file']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], p
    old = openpyxl.load_workbook(OLD / 'A100_LG系统时间拆分与扣除对照_20261025.xlsx', data_only=True)
    new = openpyxl.load_workbook(BOOK, data_only=True)
    formulas = openpyxl.load_workbook(BOOK, data_only=False)
    preserved = 0
    for s in old:
        for row in s:
            for cell in row:
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    near(new[s.title][cell.coordinate].value, cell.value)
                    preserved += 1
    assert new.sheetnames == ['先读中文解释', '视频并行与单帧', *old.sheetnames]
    stage_sheet = new['LG逐阶段明细']
    for i in range(6, 336):
        assert stage_sheet.cell(i, 2).value and stage_sheet.cell(i, 10).value
        assert stage_sheet.cell(i, 11).value == old['LG逐阶段明细'].cell(i, 2).value
        assert stage_sheet.cell(i, 12).value == old['LG逐阶段明细'].cell(i, 3).value
    for row in new['Baseline时间拆分'].iter_rows(min_row=6):
        if row[0].value:
            assert row[6].value and row[7].value
    data = json.loads((OLD / 'collected_data_and_audit.json').read_text(encoding='utf-8'))
    parallel = json.loads((NEW / 'parallel_and_loading_evidence.json').read_text(encoding='utf-8'))
    assert parallel['metadata_folders_checked'] == 90
    assert not parallel['isolated_load_duration_fields']
    repo = HERE.parents[3]
    for relative, entry in parallel['source_files'].items():
        p = repo / relative
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
    batches = new['视频并行与单帧']
    for index, key in enumerate(('01a|ours', '01a|qwen', '01b|ours', '01b|qwen'), 6):
        records = [r for r in data['mainRuns'] if r['key'] == key]
        assert len(records) == 3
        total = sum(r['seconds200'] for r in records) * 1000
        fields = 39 if key == '01a|ours' else 600
        slots = 624 if key == '01a|ours' else 600
        near(batches.cell(index, 8).value, total)
        near(batches.cell(index, 9).value, total / fields)
        near(batches.cell(index, 10).value, total / 600)
        near(batches.cell(index, 11).value, total / 2400)
        near(batches.cell(index, 12).value, total / (slots * 4))
        assert batches.cell(index, 3).value == fields
    temporal = [r for r in parallel['fields'] if r['key'] == '01a|ours']
    assert len(temporal) == 39 and sum(r['valid'] for r in temporal) == 600
    assert all(r['slots'] == 16 and r['frames'] == 64 and r['output_shape'] == [1, 16] for r in temporal)
    assert sum(r['padding'] for r in temporal) == 24
    assert len([r for r in parallel['fields'] if r['key'] == '01b|ours']) == 600
    for r in temporal:
        near(r['total_ms'], sum(r[k] for k in ('frontend_ms', 'body_ms', 'gap_ms', 'fixed_ms')))
    near(batches['K7'].value / batches['K6'].value, batches['J7'].value / batches['J6'].value)
    near(batches['K9'].value / batches['K8'].value, batches['J9'].value / batches['J8'].value)
    count = 0
    for sheet in formulas:
        for row in sheet:
            for cell in row:
                if cell.data_type == 'f':
                    count += 1
                    assert new[sheet.title][cell.coordinate].value is not None
                assert new[sheet.title][cell.coordinate].data_type != 'e', (sheet.title, cell.coordinate)
    report = dict(status='passed', no_models_no_GPU=True, preserved_numeric_cells=preserved,
                  stage_rows_with_Chinese_notes=330, formulas=count,
                  temporal_fields=39, temporal_executed_slots=624, temporal_valid_videos=600,
                  temporal_frames_per_field=64, spatial_original_videos_per_field=1,
                  isolated_model_loading_time='not separately recorded',
                  original_delivery_manifest_sha256=sha(OLD / 'delivery_manifest.json'))
    audit = NEW / 'independent_annotation_audit.json'
    assert not (NEW / 'delivery_manifest.json').exists(), 'Preserve sealed explanatory edition'
    audit.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if '--visual-review-accepted' in sys.argv:
        files = [BOOK, NEW / 'parallel_and_loading_evidence.json', NEW / 'lg_time_breakdown.json', audit]
        manifest = dict(status='EXPLANATORY_EDITION_ORIGINAL_TIMING_PRESERVED',
                        report=report, visually_checked_sheets=new.sheetnames,
                        outputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
        (NEW / 'delivery_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
