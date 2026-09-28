"""Capture a selected ABO checkpoint on the DVP8um bench, one full layer at a time.

Run only in an isolated bench project containing assets/best.pt, standalone/
and abo_full_query_flow_snapshot_20260928.py. The six-stage ideal bridge must
pass before the hardware SDK opens. An interrupted run reuses only complete
same-contract CCD/receipt pairs; it never overwrites a partial batch.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor


STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
BASE = Path('E:/code/guest/2026OpticsMoE/ABO_I2I_Lab_DVP_8um')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--mode', choices=('selftest', 'pilot', 'full'), default='full')
    args = parser.parse_args()
    root = args.project.resolve()
    if not root.is_dir() or not (root / 'standalone').is_dir():
        raise ValueError('Isolated project and standalone source required')
    weight = root / 'assets/best.pt'
    if digest(weight) != args.checkpoint_sha256:
        raise ValueError('Checkpoint SHA mismatch')
    sys.path[:0] = [str(root), str(BASE / 'lab_dvp8um')]
    from standalone.model import OpticalRetrieval
    from standalone.bounded_export import stage_active, quantize
    import four_image_flow as flow
    import abo_full_query_flow_snapshot_20260928 as pipeline
    from shs_physical2400 import SHSBench

    payload = torch.load(weight, map_location='cpu', weights_only=True)
    model = OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False).to('cpu')
    processor = AutoProcessor.from_pretrained(str(BASE / 'assets/processor'), local_files_only=True)
    geometry = json.loads((BASE / 'runs/abo_i2i_20260926/shs_geometry.json').read_text(encoding='utf8'))
    flow.BASE_CORNERS = np.asarray(geometry['base_corners_screen_TL_TR_BR_BL'], np.float32)
    pipeline.STAGE_CALIBRATION.clear()
    pipeline.STAGE_CALIBRATION.update({stage: (r['phase_candidate'], r['camera_orientation'])
                                       for stage, r in geometry['stage_calibration'].items()})
    pipeline.stage_active = stage_active
    protocol = json.loads((BASE / 'protocol.json').read_text(encoding='utf8'))
    gallery = [r for r in protocol['rows'] if r['split'] == 'train']
    query = [r for r in protocol['rows'] if r['split'] == 'query']
    all_rows = gallery + query
    if len(gallery) != 1600 or len(query) != 800 or len({r['sample_id'] for r in all_rows}) != 2400:
        raise ValueError('Physical gallery/query identity contract changed')
    rows = gallery[:4] if args.mode != 'full' else all_rows
    run_name = 'pilot4_selected_20260929' if args.mode != 'full' else 'layerwise_selected2400_20260929'
    out = root / 'runs' / run_name
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    if report_path.exists():
        raise FileExistsError('Complete run exists; refusing restart')

    phase_dir = out / 'phase'
    phase_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        phases = flow.phase_planes(model)
    phase_paths = {}
    for stage, radians in phases.items():
        path = phase_dir / (stage + '.bmp')
        expected = pipeline.selected_phase(radians, pipeline.STAGE_CALIBRATION[stage][0])
        if path.exists():
            if not np.array_equal(np.array(Image.open(path)), expected):
                raise ValueError(f'Existing phase changed: {stage}')
        else:
            Image.fromarray(expected).save(path)
        phase_paths[stage] = path
    if set(phase_paths) != set(STAGES):
        raise ValueError('Expected exactly six phase planes')
    contract = dict(checkpoint_sha256=args.checkpoint_sha256, phase_sha256={
        stage: digest(path) for stage, path in phase_paths.items()},
        encoding='bounded tanh(abs/.5) before phase; round255a without peak scaling',
        exposure_us=400, gain='Gain_X4', wait_ms=240,
        geometry=geometry['base_corners_screen_TL_TR_BR_BL'],
        stage_orientation=pipeline.STAGE_CALIBRATION,
        ids=[r['sample_id'] for r in rows])
    cp = out / 'contract.json'
    normalized = json.loads(json.dumps(contract))
    if cp.exists():
        if json.loads(cp.read_text(encoding='utf8')) != normalized:
            raise ValueError('Resume contract changed')
    else:
        write_json(cp, contract)

    counts = {stage: 0 for stage in STAGES}
    current = [None]
    warmed = set()
    started = time.perf_counter()

    def read(stage, ids):
        arrays = []
        folder = out / 'ccd' / stage
        for sid in ids:
            image_path, receipt_path = folder / (sid + '.png'), folder / (sid + '.json')
            if not image_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(f'Incomplete CCD pair: {stage}/{sid}')
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            if receipt['phase_sha256'] != digest(phase_paths[stage]) or receipt['wait_ms'] != 240:
                raise ValueError(f'CCD contract mismatch: {stage}/{sid}')
            if receipt['exposure']['exposure_us'] != 400:
                raise ValueError(f'Exposure mismatch: {stage}/{sid}')
            array = np.array(Image.open(image_path))
            if np.percentile(array, 99) < 15:
                raise RuntimeError(f'Dark CCD preserved for diagnosis: {stage}/{sid}')
            arrays.append(array)
        return np.stack(arrays)

    class EndCurrentLayer(Exception):
        pass

    def capture(bench, unused, stage, phase_path, active, ids, orientation):
        if STAGES.index(stage) > STAGES.index(current[0]):
            raise EndCurrentLayer()
        folder = out / 'ccd' / stage
        missing = []
        for j, sid in enumerate(ids):
            image_exists = (folder / (sid + '.png')).is_file()
            receipt_exists = (folder / (sid + '.json')).is_file()
            if image_exists != receipt_exists:
                raise RuntimeError(f'Partial CCD pair needs audit: {stage}/{sid}')
            if not image_exists:
                missing.append(j)
        if stage == current[0] and missing:
            selected = active[missing]
            amp_dir = out / 'amplitude' / stage
            amp_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            selected_ids = [ids[j] for j in missing]
            for sid, bitmap in zip(selected_ids, quantize(selected)):
                path = amp_dir / (sid + '.bmp')
                Image.fromarray(flow.active_to_native(bitmap)).save(path)
                paths.append(path)
            if stage not in warmed:
                for rep in range(2):
                    bench.capture('warmup_' + stage, phase_path, [paths[0]],
                                  [selected_ids[0] + f'_warmup{rep}'], orientation, save=True)
                warmed.add(stage)
            bench.capture(stage, phase_path, paths, selected_ids, orientation, save=True)
            for path in paths:
                path.unlink()
        raw = read(stage, ids)
        return torch.from_numpy(raw).to('cpu').float(), {'reused_or_captured': True}, 1.

    first = rows[:4]
    images = [flow.picture(BASE / 'data' / row['image_path'],
                           model.metadata.get('input_preprocessing', 'contain_white')) for row in first]
    batch = flow.inputs(processor, images, torch.device('cpu'))
    model.vision.optics.router.measured_ccd = None
    model.language.optics.router.measured_ccd = None
    with torch.inference_mode():
        ideal = flow.snapshot_simulation(model, batch)

    def ideal_capture(bench, output, stage, phase, active, ids, orientation):
        return ideal[stage].to('cpu').float(), {}, 1.

    pipeline.capture_stage = ideal_capture
    with torch.inference_mode():
        replay = pipeline.process_batch(model, processor, None, out, phase_paths, first)
    error = float((replay['descriptor'] - ideal['descriptor']).abs().max())
    write_json(out / 'bridge_selftest.json', dict(maximum_descriptor_error=error, samples=4, camera_used=False))
    if error >= 1e-5:
        raise RuntimeError('Ideal six-stage bridge mismatch')
    if args.mode == 'selftest':
        print(json.dumps(dict(status='selftest_complete', bridge_max_error=error)), flush=True)
        return
    pipeline.capture_stage = capture
    chunks = out / 'features'
    chunks.mkdir(exist_ok=True)
    with SHSBench(out, 400, 240, phase_paths) as bench:
        for stage in STAGES:
            current[0] = stage
            for index in range(0, len(rows), 4):
                sample = rows[index:index + 4]
                with torch.inference_mode():
                    try:
                        result = pipeline.process_batch(model, processor, bench, out, phase_paths, sample)
                    except EndCurrentLayer:
                        result = None
                if stage == STAGES[-1]:
                    if result is None:
                        raise RuntimeError('Last layer did not yield descriptors')
                    torch.save(result, chunks / f'{index:06d}.pt')
                counts[stage] = index + len(sample)
                progress = dict(status='capturing', stage=stage, stage_completed=counts[stage],
                                stage_total=len(rows), ccd_counts=counts, total_ccd=sum(counts.values()),
                                checkpoint_sha256=args.checkpoint_sha256, elapsed_seconds=time.perf_counter()-started)
                write_json(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if args.mode == 'pilot':
        report = dict(status='complete', mode='pilot', checkpoint_sha256=args.checkpoint_sha256,
                      samples=len(rows), ccd_counts=counts, total_ccd=sum(counts.values()),
                      phase_sha256=contract['phase_sha256'], bridge_max_error=error)
        write_json(report_path, report)
        write_json(out / 'progress.json', report)
        print(json.dumps(report), flush=True)
        return
    vectors, ids = [], []
    for index in range(0, len(rows), 4):
        result = torch.load(chunks / f'{index:06d}.pt', map_location='cpu', weights_only=True)
        vectors.append(result['descriptor'])
        ids.extend(result['ids'])
    vectors = torch.cat(vectors)
    if ids != [r['sample_id'] for r in rows]:
        raise ValueError('Descriptor identities changed')
    bank = {'ids': ids, 'vectors': vectors}
    torch.save(bank, out / 'physical_features.pt')
    predictions, metrics = pipeline.evaluate(vectors[len(gallery):], query, protocol, bank)
    report = dict(status='complete', checkpoint_sha256=args.checkpoint_sha256,
                  samples=2400, gallery=1600, query=800, ccd_counts=counts,
                  phase_sha256=contract['phase_sha256'], physical_to_physical=True,
                  metrics=metrics, predictions=predictions, elapsed_seconds=time.perf_counter()-started)
    write_json(report_path, report)
    write_json(out / 'progress.json', dict(status='complete', metrics=metrics, ccd_counts=counts))
    print(json.dumps(dict(status='complete', metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
