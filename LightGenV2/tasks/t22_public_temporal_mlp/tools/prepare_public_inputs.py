"""Prepare normalized RGB TEST frames for the standalone inference release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch


def decode_frames(path):
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        raise ValueError('Cannot decode video: ' + str(path))
    frames = []
    try:
        for fraction in np.linspace(.1, .9, 4):
            index = min(count - 1, int(round(fraction * (count - 1))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if not success:
                raise ValueError('Frame decoding failed: ' + str(path))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            side = max(2, int(round(min(height, width) * .65)))
            top, left = (height - side) // 2, (width - side) // 2
            cropped = rgb[top:top + side, left:left + side]
            resized = cv2.resize(cropped, (448, 448), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(resized.copy()).permute(2, 0, 1))
    finally:
        capture.release()
    return torch.stack(frames)


def write_array(archive, name, value):
    with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-csv', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-scores', type=Path,
                        help='Human MOS CSV, defaulting to reference_mos.csv beside the input archive')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    scores_path = args.reference_scores or args.output.with_name('reference_mos.csv')
    if scores_path.resolve() == args.output.resolve():
        raise ValueError('Video frames and reference scores require separate output paths')
    torch.set_num_threads(4)
    with args.evaluation_csv.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row['sample_id'] for row in rows]
    if len(ids) != 558 or len(set(ids)) != 558:
        raise ValueError('Expected the selected 558 distinct TEST sample IDs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name('.' + args.output.name + '.building')
    identities = {}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        metadata = {
            'format_version': 2, 'sample_ids': np.asarray(ids),
            'normalization': '(RGB / 255 - 0.5) / 0.5',
            'storage_precision': 'float16 preserving bfloat16 patch-projection inputs',
            'frame_fractions': np.linspace(.1, .9, 4), 'crop_fraction': .65,
            'source': 'LGVQ video frames',
        }
        for name, value in metadata.items():
            write_array(archive, name, value)
        for field, start in enumerate(range(0, len(rows), 16), 1):
            paths = [args.video_root / row['video_path'] for row in rows[start:start + 16]]
            pixels = torch.stack([decode_frames(path) for path in paths])
            normalized = (pixels.float() / 255.0 - .5) / .5
            projection_input = normalized.to(torch.bfloat16)
            stored = projection_input.to(torch.float16)
            if not torch.equal(stored.to(torch.bfloat16), projection_input):
                raise ValueError('Stored RGB values changed the patch-projection input')
            recovered_pixels = ((stored.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            if not torch.equal(recovered_pixels, pixels):
                raise ValueError('Preview RGB recovery differs from decoded pixels')
            write_array(archive, f'rgb_{field:04d}', stored.numpy())
            for row, path in zip(rows[start:start + 16], paths):
                identities[row['sample_id']] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f'Packed {min(start + 16, len(rows))}/{len(rows)} normalized videos', flush=True)
    with np.load(temporary, allow_pickle=False) as archive:
        assert archive['sample_ids'].tolist() == ids
        assert archive['rgb_0035'].shape == (14, 4, 3, 448, 448)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_scores = scores_path.with_name('.' + scores_path.name + '.building')
    with temporary_scores.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_id', 'target_MOS'))
        writer.writerows((row['sample_id'], float(row['mos'])) for row in rows)
    temporary_scores.replace(scores_path)
    temporary.replace(args.output)
    receipt = {'videos': 558, 'unique_video_hashes': len(set(identities.values())),
               'fields': 35, 'stored_padding_videos': 0,
               'maximum_patch_input_difference': 0.0, 'recovered_preview_pixels_exact': True,
               'source_video_sha256': identities,
               'npz_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'reference_scores_sha256': hashlib.sha256(scores_path.read_bytes()).hexdigest(),
               'npz_bytes': args.output.stat().st_size}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_video_sha256'}, indent=2))


if __name__ == '__main__':
    main()
