"""Read sealed Temporal timing CSVs and audit the new 16-video workbook only."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE / 'outputs/01a061e2-6f88-7913-87f5-b632000e4c83'
SEALED = BASE / 'complete_20261025_01'
OUT = BASE / 'temporal_gpu16_20261025_01'
BOOK = OUT / 'A100_Temporal_16视频_LG显卡时间对完整Qwen_20261025.xlsx'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def near(a, b):
    assert isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), (a, b)


def collect():
    source = SEALED / 'collected_data_and_audit.json'
    assert sha(source) == 'f0cd430b99aae15c69ddbeeeee1aa2d2b1334a6e804d7a91097a79c6659d2ffe'
    data = json.loads(source.read_text(encoding='utf-8'))
    catalog = {r['file']: r for r in data['raw_files']}
    used = {}

    def read(folder, name, kind='json'):
        rel = (Path(folder) / name).as_posix()
        p, expected = REPO / rel, catalog[rel]
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], rel
        used[rel] = dict(bytes=p.stat().st_size, sha256=expected['sha256'])
        if kind == 'csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                return list(csv.DictReader(f))
        return json.loads(p.read_text(encoding='utf-8-sig'))

    lg_runs = [r for r in data['mainRuns'] if r['key'] == '01a|ours']
    q_runs = [r for r in data['tempRuns'] if r['key'] in ('40GB|qwen|100', '40GB|qwen|50')]
    assert len(lg_runs) == 3 and len(q_runs) == 6
    points, stages, reference_ids = [], [], None
    for run in lg_runs:
        folder = run['folder']
        fields, raw = read(folder, 'fields.csv', 'csv'), read(folder, 'stages.csv', 'csv')
        identity, summary = read(folder, 'identity.json'), read(folder, 'summary.json')
        prepared = read(folder, 'prepared_inputs.json')
        ids = [sid for field in fields for sid, valid in zip(json.loads(field['source_ids']), json.loads(field['valid'])) if valid]
        assert reference_ids is None or ids == reference_ids
        reference_ids = ids
        assert len(fields) == 13 and len(ids) == 200
        assert all(json.loads(f['output_shape']) == [1, 16] for f in fields)
        assert all(x['vision_tokens']['shape'] == [1, 16, 4, 49, 1024] for x in prepared)
        assert identity['batch'] == 1
        point = dict(system='LG', cap=100, repeat=run['repeat'], valid=200, fields=13, slots=208,
                     original_ms=run['seconds200'] * 1000, folder=folder, gpu_front_event=0.,
                     gpu_body_event=0., gpu_excluded_event=0., gpu_selected_wall=0.,
                     cpu_prepare=0., cpu_output=0., unspanned_front=0.,
                     gaps=sum(float(f['internal_gap_ms']) for f in fields), fixed=13 * 6 * 1.0447)
        for row in raw:
            wall = float(row['synchronized_wall_ms'])
            near(wall, (float(row['end_monotonic']) - float(row['start_monotonic'])) * 1000)
            event = None if row['cuda_event_ms'] == '' else float(row['cuda_event_ms'])
            charged = row['charged'] == 'True'
            stage = dict(system='LG', repeat=run['repeat'], field=int(row['sample_index']),
                         stage=row['stage'], family=row['family'], region=row['region'], charged=charged,
                         wall_ms=wall, event_ms=event, folder=folder)
            stages.append(stage)
            if event is not None:
                if charged:
                    point['gpu_front_event' if row['region'] == 'frontend' else 'gpu_body_event'] += event
                    point['gpu_selected_wall'] += wall
                else:
                    point['gpu_excluded_event'] += event
            elif charged:
                point['cpu_prepare' if row['region'] == 'frontend' else 'cpu_output'] += wall
        front = sum(float(f['frontend_wall_ms']) for f in fields)
        point['unspanned_front'] = front - sum(s['wall_ms'] for s in stages if s['repeat'] == run['repeat'] and s['region'] == 'frontend')
        assert point['unspanned_front'] >= 0
        assert all(sum(s['family'] == 'optical_simulation_FFT' for s in stages if s['repeat'] == run['repeat'] and s['field'] == i) == 6 for i in range(13))
        near(point['original_ms'], sum(point[k] for k in ('cpu_prepare', 'gpu_selected_wall', 'cpu_output', 'unspanned_front', 'gaps', 'fixed')))
        near(summary['seconds200'] * 1000, point['original_ms'])
        points.append(point)
    for run in q_runs:
        folder = run['folder']
        calls, identity = read(folder, 'calls.csv', 'csv'), read(folder, 'identity.json')
        assert len(calls) == 200 and [c['sample_id'] for c in calls] == reference_ids
        assert identity['batch'] == 1 and all(len(json.loads(c['frame_positions'])) == 4 for c in calls)
        pieces = defaultdict(float)
        for c in calls:
            stamps = [float(c[k]) for k in ('preprocess_start', 'prepare_end', 'model_call_start', 'first_block_start', 'gpu_output', 'call_return')]
            assert all(b >= a for a, b in zip(stamps, stamps[1:]))
            for name, a, b in zip(('cpu_prepare', 'transfer', 'model_front', 'body_head', 'cpu_output'), stamps, stamps[1:]):
                pieces[name] += (b - a) * 1000
        gaps = sum((float(b['preprocess_start']) - float(a['call_return'])) * 1000 for a, b in zip(calls, calls[1:]))
        total = (float(calls[-1]['call_return']) - float(calls[0]['preprocess_start'])) * 1000
        near(total, sum(pieces.values()) + gaps)
        near(total, run['seconds200'] * 1000)
        points.append(dict(system='Qwen', cap=int(run['key'].split('|')[-1]), repeat=run['repeat'],
                           valid=200, fields=200, slots=200, original_ms=total, gaps=gaps, folder=folder, **pieces))
    assert len(points) == 9
    groups = {}
    for s in stages:
        key = s['stage']
        if key not in groups:
            groups[key] = dict(stage=key, family=s['family'], region=s['region'], charged=s['charged'], wall_ms=0., event_ms=0., event_count=0, calls=0)
        g = groups[key]
        assert (g['family'], g['region'], g['charged']) == (s['family'], s['region'], s['charged'])
        g['wall_ms'] += s['wall_ms']; g['calls'] += 1
        if s['event_ms'] is not None:
            g['event_ms'] += s['event_ms']; g['event_count'] += 1
    return dict(points=points, stages=list(groups.values()), source_files=used, no_models_no_GPU=True,
                source_sha256=sha(source), compared_valid_ids=reference_ids,
                scope='Temporal only; LG100 GPU events versus full Qwen40GB MPS100 and50',
                normalization='sum three rounds /600 valid videos *16; retains LG padding cost')


def audit():
    import openpyxl
    assert not (OUT / 'delivery_manifest.json').exists(), 'Preserve sealed report'
    data = collect()
    book = openpyxl.load_workbook(BOOK, data_only=True)
    formulas = openpyxl.load_workbook(BOOK, data_only=False)
    lg = [r for r in data['points'] if r['system'] == 'LG']
    total = lambda key: sum(r[key] for r in lg) * 16 / 600
    selected = total('gpu_front_event') + total('gpu_body_event')
    all_gpu, plus = selected + total('gpu_excluded_event'), selected + total('fixed')
    q = {cap: sum(r['original_ms'] for r in data['points'] if r['system'] == 'Qwen' and r['cap'] == cap) * 16 / 600 for cap in (100, 50)}
    s = book['16视频对照']
    for row, expected in ((6, all_gpu), (7, selected), (8, plus), (9, q[100]), (10, q[50]), (11, total('original_ms'))):
        near(s.cell(row, 2).value, expected)
        near(s.cell(row, 3).value, q[100] / expected)
        near(s.cell(row, 4).value, q[50] / expected)
    near(s['B22'].value, selected)
    near(sum(s.cell(row, 2).value for row in range(16, 22)), selected)
    near(s['B38'].value, total('original_ms'))
    near(sum(s.cell(row, 2).value for row in range(28, 38)), total('original_ms'))
    for cap, column in ((100, 2), (50, 3)):
        near(sum(s.cell(row, column).value for row in range(43, 49)), q[cap])
        near(s.cell(49, column).value, q[cap])
    count = 0
    for sheet in formulas:
        for row in sheet:
            for cell in row:
                cached = book[sheet.title][cell.coordinate]
                assert cached.data_type != 'e', (sheet.title, cell.coordinate)
                if cell.data_type == 'f':
                    count += 1
                    assert cached.value is not None
    report = dict(status='passed', no_models_no_GPU=True, matched_valid_videos_per_configuration=600,
                  LG_fields=39, LG_executed_video_slots=624, Qwen100_calls=600, Qwen50_calls=600,
                  gpu_all_recorded_16_ms=all_gpu, gpu_selected_16_ms=selected,
                  gpu_selected_plus_fixed_16_ms=plus, Qwen_complete16_ms=q, formulas=count,
                  formula_errors=0, original_stage_files_rechecked=True,
                  comparison_is_asymmetric=True, GPU_event_coverage='recorded mutually exclusive scopes, not full hardware kernel trace')
    audit_path = OUT / 'independent_audit.json'
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if '--visual-review-accepted' in sys.argv:
        files = [BOOK, OUT / 'analysis_data.json', audit_path]
        manifest = dict(report=report, visual_review_sheets=book.sheetnames,
                        outputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
        (OUT / 'delivery_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    if '--data' in sys.argv:
        print(json.dumps(collect(), ensure_ascii=True))
    else:
        audit()
