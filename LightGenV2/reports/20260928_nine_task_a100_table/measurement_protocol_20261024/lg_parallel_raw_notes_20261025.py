"""File-only batch/frame evidence for the Chinese timing annotations. No models."""
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def collect(repo, source):
    repo, source = Path(repo), Path(source)
    data = json.loads(source.read_text(encoding='utf-8'))
    catalog = {r['file']: r for r in data['raw_files']}
    used = {}

    def read(relative, kind='json'):
        relative = str(relative).replace('\\', '/')
        p = repo / relative
        sealed = catalog[relative]
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == sealed['bytes'] and digest == sealed['sha256'], relative
        used[relative] = {'bytes': len(raw), 'sha256': digest}
        if kind == 'csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                return list(csv.DictReader(f))
        return json.loads(raw.decode('utf-8-sig'))

    def near(a, b):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), (a, b)

    batch_runs, field_rows = [], []
    for run in data['mainRuns']:
        if run['key'] not in ('01a|ours', '01b|ours', '01a|qwen', '01b|qwen'):
            continue
        folder = Path(run['folder'])
        identity = read(folder / 'identity.json')
        temporal = run['key'].startswith('01a')
        if run['system'] == 'ours':
            fields = read(folder / 'fields.csv', 'csv')
            stages = read(folder / 'stages.csv', 'csv')
            prepared = read(folder / 'prepared_inputs.json')
            summary = read(folder / 'summary.json')
            slots = 16 if temporal else 1
            assert len(fields) == (13 if temporal else 200)
            assert len(prepared) == len(fields)
            for index, (field, item) in enumerate(zip(fields, prepared)):
                valid = json.loads(field['valid'])
                shape = item['vision_tokens']['shape']
                output = json.loads(field['output_shape'])
                assert len(valid) == slots
                assert shape == ([1, 16, 4, 49, 1024] if temporal else [1, 4, 196, 1024])
                assert output == ([1, 16] if temporal else [1])
                subset = [s for s in stages if int(s['sample_index']) == index]
                assert sum(s['family'] == 'optical_simulation_FFT' for s in subset) == 6
                body = sum(float(s['synchronized_wall_ms']) for s in subset
                           if s['region'] == 'body' and s['charged'] == 'True')
                fixed = 6 * 1.0447
                total = float(field['frontend_wall_ms']) + body + float(field['internal_gap_ms']) + fixed
                field_rows.append(dict(key=run['key'], repeat=run['repeat'], field=index,
                    slots=slots, frames=slots * 4, valid=sum(valid), padding=slots - sum(valid),
                    frontend_ms=float(field['frontend_wall_ms']), body_ms=body,
                    gap_ms=float(field['internal_gap_ms']), fixed_ms=fixed, total_ms=total,
                    output_shape=output, input_shape=shape, folder=run['folder']))
            selected = [r for r in field_rows if r['key'] == run['key'] and r['repeat'] == run['repeat']]
            near(sum(r['total_ms'] for r in selected), run['seconds200'] * 1000)
            assert sum(r['valid'] for r in selected) == 200
            batch_runs.append(dict(key=run['key'], repeat=run['repeat'], fields=len(fields),
                videos_per_field=slots, frames_per_field=slots * 4, valid_videos=200,
                executed_slots=sum(r['slots'] for r in selected), total_ms=run['seconds200'] * 1000,
                output_shape=selected[0]['output_shape'], input_shape=selected[0]['input_shape'],
                batch_outer=identity['batch'], folder=run['folder']))
            near(summary['sample_ms'], run['seconds200'] * 1000 / 200)
        else:
            calls = read(folder / 'calls.csv', 'csv')
            assert identity['batch'] == 1 and len(calls) == 200
            assert all(len(json.loads(c['frame_positions'])) == 4 for c in calls)
            batch_runs.append(dict(key=run['key'], repeat=run['repeat'], fields=200,
                videos_per_field=1, frames_per_field=4, valid_videos=200, executed_slots=200,
                total_ms=run['seconds200'] * 1000, batch_outer=1, folder=run['folder']))

    # Look only for explicitly named load duration fields. Clock anchors or a
    # loading status are not measurements of isolated model loading time.
    def duration_fields(obj, prefix=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                name = prefix + '.' + key
                if re.search(r'(model_?load|loading|startup).*(seconds|duration|elapsed)', key, re.I):
                    yield name, value
                if key not in ('sources', 'source_blobs', 'input_files', 'video_metadata_before',
                               'input_files_metadata_before', 'command'):
                    yield from duration_fields(value, name)
        elif isinstance(obj, list):
            for value in obj:
                yield from duration_fields(value, prefix + '[]')

    load_fields = []
    seen = set()
    for run in data['mainRuns'] + data['tempRuns']:
        if run['folder'] in seen:
            continue
        seen.add(run['folder'])
        for name in ('identity.json', 'report.json', 'summary.json', 'status.json', 'process_audit.json'):
            rel = (Path(run['folder']) / name).as_posix()
            if rel not in catalog:
                continue
            for key, value in duration_fields(read(rel)):
                load_fields.append(dict(folder=run['folder'], file=name, key=key, value=value))
    return dict(no_models_no_GPU=True, batch_runs=batch_runs, fields=field_rows,
                isolated_load_duration_fields=load_fields, metadata_folders_checked=len(seen),
                source_files=used)


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1], sys.argv[2]), ensure_ascii=True))
