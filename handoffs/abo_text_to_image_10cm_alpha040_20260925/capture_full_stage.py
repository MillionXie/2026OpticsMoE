"""Resumable DVP/Holoeye/Meadowlark stage capture for pinned T08 TEST.

No simulation PCC gate. The capture is canonical ROI uint8, with no pixel-value
normalization. Phase and amplitude are driven in the logged-in desktop session.
"""
from __future__ import annotations

import argparse
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
sys.path[:0] = [str(ROOT / 'ABO_I2I_Lab_DVP_8um'),
                str(ROOT / 'ABO_Lab_SHS_8um'),
                str(ROOT / 'ABO_I2I_DVP_adapter_20260922')]
from lab_dvp8um import four_image_flow as flow  # noqa: E402

STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
ORIENTATIONS = ('flip_v', 'identity', 'identity', 'flip_v', 'rot270', 'rot270')
EXPECTED = 'cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'
CORNER_POINTS = [[995, 172], [4310, 172], [4303, 3440], [980, 3435]]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def native_bmp(compact, output):
    gray = np.asarray(Image.open(compact).convert('L'))
    side = gray.shape[0]
    if gray.shape != (side, side) or side not in (224, 478):
        raise RuntimeError(f'Unexpected logical amplitude {gray.shape}: {compact}')
    native_side = round(side * 17 / 8)
    positions = (np.arange(native_side) + .5 - native_side / 2) * 8
    idx = np.floor(positions / 17 + side / 2).astype(int).clip(0, side - 1)
    square = gray[np.ix_(idx, idx)]
    canvas = np.zeros((1080, 1920), dtype=np.uint8)
    top, left = (1080 - native_side) // 2, (1920 - native_side) // 2
    canvas[top:top + native_side, left:left + native_side] = square
    Image.fromarray(canvas).save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--exposure-us', type=float, default=10000)
    parser.add_argument('--wait-ms', type=float, default=240)
    parser.add_argument('--discard-frames', type=int, default=6)
    args = parser.parse_args()
    if args.exposure_us <= 0 or args.wait_ms < 0 or args.discard_frames < 1:
        raise ValueError('Invalid capture timing')
    index = STAGES.index(args.stage)
    stage = PROJECT / 'full_test' / f'{index + 1:02d}_{args.stage}'
    compact = stage / 'compact_amplitude'
    captured = stage / 'ccd_captured'
    phase = PROJECT / 'phase_bmp_provisional' / f'{args.stage}.bmp'
    lines = (stage / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
    rows = {row['key']: row for row in map(json.loads, lines)}
    expected = 2400 if index < 3 else 2500
    if len(rows) != expected:
        raise RuntimeError(f'Incomplete {args.stage} manifest: {len(rows)}/{expected}')
    if any(row['checkpoint_sha256'] != EXPECTED for row in rows.values()):
        raise RuntimeError('Manifest checkpoint mismatch')
    if not phase.is_file():
        raise FileNotFoundError(phase)
    flow.BASE_CORNERS = np.float32(CORNER_POINTS)
    captured.mkdir(exist_ok=True)
    temp = stage / '_active_amplitude.bmp'
    journal = stage / 'capture_journal.jsonl'
    with flow.Bench(stage, args.exposure_us, args.wait_ms,
                    {args.stage: phase}) as bench:
        receipt = bench.phase.show(phase)
        print('phase_receipt', receipt, 'phase_sha256', digest(phase), flush=True)
        print('camera_settings', bench.settings, 'roi', CORNER_POINTS,
              'orientation', ORIENTATIONS[index], flush=True)
        for number, (key, row) in enumerate(rows.items(), 1):
            target = captured / f'{key}.png'
            if target.is_file():
                image = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
                if image is not None and image.shape == (478, 478):
                    continue
                raise RuntimeError(f'Invalid existing CCD image: {target}')
            source = compact / row['file']
            if digest(source) != row['sha256']:
                raise RuntimeError(f'Amplitude hash mismatch: {source}')
            native_bmp(source, temp)
            bench.amp.preload_files([temp])
            bench.amp.display_file(temp)
            time.sleep(args.wait_ms / 1000)
            frame_ids = []
            for _ in range(args.discard_frames):
                raw, meta = bench.camera.capture()
                frame_ids.append(meta['frame_id'])
            canonical = flow.orient(flow.warp(raw), ORIENTATIONS[index])
            if canonical.shape != (478, 478):
                raise RuntimeError(f'Bad CCD shape: {canonical.shape}')
            saturated = float(np.mean(canonical >= 255))
            stats = {'stage': args.stage, 'key': key, 'frame_ids': frame_ids,
                     'mean': float(canonical.mean()),
                     'p99': float(np.percentile(canonical, 99)),
                     'maximum': int(canonical.max()),
                     'saturation_fraction': saturated,
                     'exposure_us': args.exposure_us, 'wait_ms': args.wait_ms,
                     'phase_sha256': digest(phase),
                     'amplitude_sha256': row['sha256'],
                     'roi_full_sensor_tl_tr_br_bl': CORNER_POINTS,
                     'orientation': ORIENTATIONS[index],
                     'photometric_normalization': 'none'}
            if saturated > .01:
                raise RuntimeError(f'Frame {number} saturated: {saturated:.3%}')
            Image.fromarray(canonical).save(target)
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(stats, ensure_ascii=False) + '\n')
            if number <= 4 or number % 20 == 0:
                print(f'captured {number}/{expected} {key} p99={stats["p99"]:.1f} '
                      f'sat={saturated:.4%}', flush=True)
    print(f'{args.stage.upper()}_COMPLETE {expected}/{expected}', flush=True)


if __name__ == '__main__':
    main()
