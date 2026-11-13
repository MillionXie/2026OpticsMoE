"""Predict one quality score per video with the LightGen2 checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .model import load_lightgen2
else:
    from model import load_lightgen2


PROJECT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}


def frames_from_video(path):
    """Sample four RGB frames with the checkpoint's fixed crop and resize."""
    import cv2

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
            top = (height - side) // 2
            left = (width - side) // 2
            cropped = rgb[top:top + side, left:left + side]
            resized = cv2.resize(cropped, (448, 448), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(resized.copy()).permute(2, 0, 1))
    finally:
        capture.release()
    return torch.stack(frames)


def videos_in_directory(directory):
    """Return supported videos in filename order, without recursive discovery."""
    paths = sorted(path for path in Path(directory).iterdir()
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise ValueError('No supported videos in ' + str(directory))
    return paths


def read_evaluation_csv(path, video_root, *, limit=None):
    """Preserve the evaluation file's sample order and its temporal MOS labels."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if limit is not None:
        rows = rows[:limit]
    if len(rows) < 2 or not {'sample_id', 'video_path', 'mos'}.issubset(rows[0]):
        raise ValueError('Evaluation CSV requires sample_id, video_path and mos columns')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Evaluation sample IDs must be unique')
    targets = np.asarray([float(row['mos']) for row in rows])
    if not np.isfinite(targets).all():
        raise ValueError('Evaluation MOS labels must be finite')
    paths = [Path(video_root) / row['video_path'] for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Evaluation video is missing: ' + missing[0])
    return rows, paths, targets


def iter_video_fields(video_paths):
    """Group 16 ordered videos and cyclically pad the final optical field."""
    if not video_paths:
        raise ValueError('At least one video is required')
    first_frames = []
    for start in range(0, len(video_paths), 16):
        chunk = video_paths[start:start + 16]
        frames = [frames_from_video(path) for path in chunk]
        if not first_frames:
            first_frames = frames[:15]
        padding = [first_frames[index % len(first_frames)]
                   for index in range(16 - len(frames))]
        yield chunk, torch.stack(frames + padding)[None]


def predict_videos(model, video_paths, device, *, show_progress=False):
    records = []
    for field_index, (chunk, pixels) in enumerate(iter_video_fields(video_paths)):
        output = model(pixels=pixels.to(device))
        prediction = output['prediction'][0]
        records.extend({'video': str(path), 'field': field_index, 'slot': slot,
                        'MOS': float(prediction[slot])}
                       for slot, path in enumerate(chunk))
        if show_progress:
            print(f'Processed {len(records)}/{len(video_paths)} videos', flush=True)
    return records


def save_predictions(path, records, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model': 'LightGen2', 'predictions': records}
    if metrics is not None:
        payload['metrics'] = metrics
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)
                    + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'weights/LightGen2.pt')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--videos', type=Path, nargs='+', help='Videos in optical-slot order')
    inputs.add_argument('--video-dir', type=Path, help='Read supported videos in filename order')
    inputs.add_argument('--evaluation-csv', type=Path, help='Ordered video paths and MOS labels')
    parser.add_argument('--video-root', type=Path, help='Base directory for evaluation video paths')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/predictions.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluation = None
    if args.evaluation_csv is not None:
        if args.video_root is None:
            parser.error('--evaluation-csv requires --video-root')
        evaluation, paths, targets = read_evaluation_csv(args.evaluation_csv, args.video_root)
    else:
        paths = args.videos if args.videos is not None else videos_in_directory(args.video_dir)
    model, _ = load_lightgen2(args.weights, args.device)
    records = predict_videos(model, paths, args.device, show_progress=True)
    metrics = None
    if evaluation is not None:
        from scipy.stats import spearmanr
        predictions = np.asarray([record['MOS'] for record in records])
        correlation = None
        if np.ptp(targets) > 0 and np.ptp(predictions) > 0:
            correlation = float(spearmanr(predictions, targets).statistic)
        for record, row, target in zip(records, evaluation, targets):
            record.update(sample_id=row['sample_id'], target_MOS=float(target))
        metrics = {'n': len(records), 'SRCC': correlation}
    save_predictions(args.output, records, metrics)
    print(json.dumps(metrics if metrics is not None else records, indent=2))


if __name__ == '__main__':
    main()
