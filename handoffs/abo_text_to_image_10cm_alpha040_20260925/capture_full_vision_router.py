"""Capture all T08 TEST vision-router CCDs in the interactive Windows session.

Uses the existing 2026-09-24 ROI/orientation and historical phase encoding.
Simulation PCC is intentionally not a capture gate.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(r'E:\code\guest\2026OpticsMoE')
PROJECT = ROOT / 'ABO_T2I_10cm_alpha040_20260925'
FLOW = ROOT / 'ABO_I2I_Lab_DVP_8um'
sys.path[:0] = [str(FLOW), str(ROOT / 'ABO_Lab_SHS_8um'),
                str(ROOT / 'ABO_I2I_DVP_adapter_20260922')]
from lab_dvp8um import four_image_flow as flow  # noqa: E402

STAGE = PROJECT / 'full_test' / '01_vision_router'
COMPACT = STAGE / 'compact_amplitude'
CAPTURE = STAGE / 'ccd_captured'
PHASE = PROJECT / 'phase_bmp_provisional' / 'vision_router.bmp'
WAIT_S = .240
EXPOSURE_US = 10000.0
EXPECTED = 'cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'
ORIENTATION = 'flip_v'
CORNER_POINTS = [[995, 172], [4310, 172], [4303, 3440], [980, 3435]]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def native_bmp(compact, output):
    gray = np.asarray(Image.open(compact).convert('L'))
    if gray.shape != (224, 224):
        raise RuntimeError(f'Expected 224x224 router amplitude: {compact}')
    size = round(224 * 17 / 8)
    positions = (np.arange(size) + .5 - size / 2) * 8
    idx = np.floor(positions / 17 + 224 / 2).astype(int).clip(0, 223)
    square = gray[np.ix_(idx, idx)]
    canvas = np.zeros((1080, 1920), dtype=np.uint8)
    y, x = (1080 - size) // 2, (1920 - size) // 2
    canvas[y:y + size, x:x + size] = square
    Image.fromarray(canvas).save(output)


def main():
    flow.BASE_CORNERS = np.float32(CORNER_POINTS)
    lines = (STAGE / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
    rows = {int(row['index']): row for row in map(json.loads, lines)}
    if len(rows) != 2400 or set(rows) != set(range(2400)):
        raise RuntimeError(f'Incomplete TEST manifest: {len(rows)}')
    if any(row['checkpoint_sha256'] != EXPECTED for row in rows.values()):
        raise RuntimeError('Mixed checkpoint in manifest')
    if not PHASE.is_file():
        raise FileNotFoundError(PHASE)
    CAPTURE.mkdir(exist_ok=True)
    temp = STAGE / '_active_amplitude.bmp'
    journal = STAGE / 'capture_journal.jsonl'
    with flow.Bench(STAGE, EXPOSURE_US, WAIT_S * 1000, {'vision_router': PHASE}) as bench:
        receipt = bench.phase.show(PHASE)
        print('phase_receipt', receipt, 'phase_sha256', digest(PHASE), flush=True)
        print('camera_settings', bench.settings, 'roi', CORNER_POINTS,
              'orientation', ORIENTATION, flush=True)
        for i in range(2400):
            row = rows[i]
            target = CAPTURE / f'image_{i:04d}.png'
            if target.is_file():
                if cv2.imread(str(target), cv2.IMREAD_UNCHANGED).shape == (478, 478):
                    continue
                raise RuntimeError(f'Invalid existing CCD image: {target}')
            source = COMPACT / row['file']
            if digest(source) != row['sha256']:
                raise RuntimeError(f'Amplitude hash mismatch: {source}')
            native_bmp(source, temp)
            bench.amp.preload_files([temp])
            bench.amp.display_file(temp)
            time.sleep(WAIT_S)
            frames, frame_ids = [], []
            for _ in range(6):
                raw, meta = bench.camera.capture()
                frames.append(raw)
                frame_ids.append(meta['frame_id'])
            canonical = flow.orient(flow.warp(frames[-1]), ORIENTATION)
            if canonical.shape != (478, 478):
                raise RuntimeError(f'Bad CCD shape: {canonical.shape}')
            saturated = float(np.mean(canonical >= 255))
            stats = {'index': i, 'sample_id': row['sample_id'], 'frame_ids': frame_ids,
                     'mean': float(canonical.mean()), 'p99': float(np.percentile(canonical, 99)),
                     'maximum': int(canonical.max()), 'saturation_fraction': saturated,
                     'exposure_us': EXPOSURE_US, 'wait_ms': WAIT_S * 1000,
                     'phase_sha256': digest(PHASE), 'amplitude_sha256': row['sha256'],
                     'roi_full_sensor_tl_tr_br_bl': CORNER_POINTS,
                     'orientation': ORIENTATION, 'photometric_normalization': 'none'}
            if saturated > .01:
                raise RuntimeError(f'Frame {i} is saturated: {saturated:.3%}')
            Image.fromarray(canonical).save(target)
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(stats, ensure_ascii=False) + '\n')
            if i < 4 or (i + 1) % 20 == 0:
                print(f'captured {i+1}/2400 p99={stats["p99"]:.1f} '
                      f'mean={stats["mean"]:.1f} sat={saturated:.4%}', flush=True)
    print('VISION_ROUTER_COMPLETE 2400/2400', flush=True)


if __name__ == '__main__':
    main()
