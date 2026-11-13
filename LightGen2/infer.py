"""Evaluate the complete video list and visualize one of its optical fields."""
from __future__ import annotations

import time

PROGRAM_STARTED = time.perf_counter()

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .model import load_lightgen2
else:
    from model import load_lightgen2


PROJECT = Path(__file__).resolve().parent


def read_input_metadata(archive):
    """Read ordered video identities and human labels from the RGB archive."""
    if int(archive['format_version']) != 1:
        raise ValueError('Unsupported normalized RGB archive version')
    sample_ids = archive['sample_ids'].tolist()
    targets = archive['target_mos']
    if not isinstance(sample_ids, list) or len(sample_ids) < 2:
        raise ValueError('The input archive requires at least two labeled videos')
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise ValueError('Sample IDs must be nonempty strings')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Evaluation sample IDs must be unique')
    if targets.shape != (len(sample_ids),) or not np.isfinite(targets).all():
        raise ValueError('Each sample requires one finite temporal MOS label')
    return sample_ids, targets


def iter_rgb_fields(archive, sample_ids):
    """Read preprocessed fields and cyclically pad the last one to 16 slots."""
    first_frames = None
    for field, start in enumerate(range(0, len(sample_ids), 16), 1):
        count = min(16, len(sample_ids) - start)
        frames = archive[f'rgb_{field:04d}']
        if frames.shape != (count, 4, 3, 448, 448) or frames.dtype != np.float16:
            raise ValueError(f'Field {field} requires float16 normalized RGB with shape [{count},4,3,448,448]')
        if not np.isfinite(frames).all() or frames.min() < -1 or frames.max() > 1:
            raise ValueError('Normalized RGB values must be finite and lie in [-1,1]')
        if first_frames is None:
            first_frames = frames[:15].copy()
        if count < 16:
            padding = first_frames[np.arange(16 - count) % len(first_frames)]
            frames = np.concatenate((frames, padding))
        yield start, count, torch.from_numpy(frames)[None]


@torch.inference_mode()
def evaluate(model, archive, sample_ids, targets, device, directory, *, visualize_field=1):
    """Run each field once and retain that same forward for optical export."""
    if __package__:
        from .visualize import capture_optical_fields, export_evaluation, export_visualizations
    else:
        from visualize import capture_optical_fields, export_evaluation, export_visualizations
    from scipy.stats import spearmanr

    field_count = math.ceil(len(sample_ids) / 16)
    if not 1 <= visualize_field <= field_count:
        raise ValueError(f'--visualize-field must be between 1 and {field_count}')
    records = []
    routing = {branch: {'selected': [], 'probabilities': []}
               for branch in ('vision', 'language')}
    optical_capture = None
    for field_number, (start, count, normalized_rgb) in enumerate(iter_rgb_fields(archive, sample_ids), 1):
        normalized_rgb = normalized_rgb.to(device)
        if field_number == visualize_field:
            result, captured = capture_optical_fields(model, normalized_rgb)
            optical_capture = (normalized_rgb, captured)
        else:
            result = model(normalized_rgb=normalized_rgb)
        prediction = result['prediction'][0, :count].cpu().tolist()
        for slot, value in enumerate(prediction, 1):
            records.append({'sample_id': sample_ids[start + slot - 1],
                            'field': field_number, 'slot': slot,
                            'target_MOS': float(targets[start + slot - 1]), 'predicted_MOS': value})
        for branch in routing:
            for key in routing[branch]:
                routing[branch][key].append(result['routing'][branch][key][0, :count].cpu().numpy())
        print(f'Field {field_number}/{field_count}  videos {len(records)}/{len(sample_ids)}', flush=True)

    predictions = np.asarray([record['predicted_MOS'] for record in records])
    correlation = None
    if np.ptp(targets) > 0 and np.ptp(predictions) > 0:
        correlation = float(spearmanr(predictions, targets).statistic)
    metrics = {'videos': len(records), 'fields': field_count, 'SRCC': correlation,
               'visualized_field': visualize_field}
    full_routing = {branch: {key: np.concatenate(values) for key, values in data.items()}
                    for branch, data in routing.items()}
    directory = Path(directory)
    export_evaluation(records, metrics, full_routing, directory)
    normalized_rgb, captured = optical_capture
    start = (visualize_field - 1) * 16
    real_count = min(16, len(sample_ids) - start)
    padding_ids = [sample_ids[index % len(sample_ids)] for index in range(16 - real_count)]
    field_ids = sample_ids[start:start + 16] + padding_ids
    field_records = [{'sample_id': sample_id, 'is_padding': slot >= real_count}
                     for slot, sample_id in enumerate(field_ids)]
    export_visualizations(model, field_records, normalized_rgb, captured, directory,
                          field_number=visualize_field, field_count=field_count)
    return records, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'weights/LightGen2.pt')
    parser.add_argument('--input', type=Path, default=PROJECT / 'data/test_rgb.npz',
                        help='Normalized RGB frames, ordered sample IDs and human MOS labels')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results', help='Result directory')
    parser.add_argument('--visualize-field', type=int, default=1,
                        help='One-based field number captured during the full inference')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if __package__:
        from .visualize import configure_figures
    else:
        from visualize import configure_figures
    configure_figures()
    device_name = (torch.cuda.get_device_name(torch.device(args.device))
                   if torch.device(args.device).type == 'cuda' else 'CPU')
    print('Running on ' + device_name, flush=True)
    model, _ = load_lightgen2(args.weights, args.device)
    with np.load(args.input, allow_pickle=False) as archive:
        sample_ids, targets = read_input_metadata(archive)
        _, metrics = evaluate(model, archive, sample_ids, targets, args.device, args.output,
                              visualize_field=args.visualize_field)
    metrics['total_runtime_seconds'] = time.perf_counter() - PROGRAM_STARTED
    metrics['device_name'] = device_name
    (args.output / '02_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)
                                              + '\n', encoding='utf-8', newline='\n')
    print(f"SRCC {metrics['SRCC']} over {metrics['videos']} distinct videos")
    print('Saved predictions and optical fields to ' + str(args.output.resolve()))
    print(f"Complete program runtime {metrics['total_runtime_seconds']:.3f} seconds")


if __name__ == '__main__':
    main()
