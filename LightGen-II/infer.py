"""Evaluate the complete video list and visualize one of its optical fields."""
from __future__ import annotations

import argparse
import csv
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
    if int(archive['format_version']) not in (1, 2):
        raise ValueError('Unsupported video-frame archive version')
    sample_ids = archive['sample_ids'].tolist()
    if not isinstance(sample_ids, list) or len(sample_ids) < 2:
        raise ValueError('The input archive requires at least two videos')
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise ValueError('Sample IDs must be nonempty strings')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Evaluation sample IDs must be unique')
    return sample_ids


def read_reference_scores(path, sample_ids):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['sample_id', 'target_MOS']:
            raise ValueError('Reference scores require sample_id and target_MOS columns')
        rows = list(reader)
    scores = {row['sample_id']: float(row['target_MOS']) for row in rows}
    if len(scores) != len(rows):
        raise ValueError('Reference score sample IDs must be unique')
    if set(scores) != set(sample_ids):
        raise ValueError('Reference scores must match the input video IDs')
    targets = np.asarray([scores[sample_id] for sample_id in sample_ids], dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError('Each sample requires one finite temporal MOS label')
    return targets


def iter_rgb_fields(archive, sample_ids):
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
    if __package__:
        from .visualize import capture_optical_fields, export_evaluation, export_visualizations
    else:
        from visualize import capture_optical_fields, export_evaluation, export_visualizations
    from scipy.stats import spearmanr

    field_count = math.ceil(len(sample_ids) / 16)
    if not 1 <= visualize_field <= field_count:
        raise ValueError(f'--visualize-field must be between 1 and {field_count}')
    records = []
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
                            'target_MOS': float(targets[start + slot - 1]), 'predicted_MOS': value})
        print(f'Field {field_number}/{field_count}  videos {len(records)}/{len(sample_ids)}', flush=True)

    predictions = np.asarray([record['predicted_MOS'] for record in records])
    correlation = None
    if np.ptp(targets) > 0 and np.ptp(predictions) > 0:
        correlation = float(spearmanr(predictions, targets).statistic)
    metrics = {'videos': len(records), 'SRCC': correlation}
    directory = Path(directory)
    export_evaluation(records, metrics, directory)
    normalized_rgb, captured = optical_capture
    start = (visualize_field - 1) * 16
    real_count = min(16, len(sample_ids) - start)
    padding_ids = [sample_ids[index % len(sample_ids)] for index in range(16 - real_count)]
    field_ids = sample_ids[start:start + 16] + padding_ids
    field_records = [{'sample_id': sample_id, 'is_padding': slot >= real_count}
                     for slot, sample_id in enumerate(field_ids)]
    export_visualizations(model, field_records, normalized_rgb, captured, directory,
                          field_number=visualize_field)
    return records, metrics


def check_output_directory(directory, reference=None):
    directory = Path(directory).resolve()
    reference = Path(reference if reference is not None else PROJECT/'reference_results').resolve()
    if directory == reference or reference in directory.parents or directory in reference.parents:
        raise ValueError('The output directory must not overlap reference_results.')
    for name in ('01_predictions.csv', '02_metrics.json'):
        if not (reference/name).is_file():
            raise FileNotFoundError('Missing supplied reference file: ' + str(reference/name))
    identities = {(path.stat().st_dev, path.stat().st_ino)
                  for path in reference.rglob('*') if path.is_file()}
    if directory.exists():
        if not directory.is_dir():
            raise ValueError('The output path must be a directory.')
        for path in directory.iterdir():
            resolved = path.resolve()
            if resolved == reference or reference in resolved.parents:
                raise ValueError('An output path points into reference_results: ' + str(path))
            if path.is_file() and (path.stat().st_dev, path.stat().st_ino) in identities:
                raise ValueError('An output file aliases a supplied reference: ' + str(path))
    return directory


def compare_reference_results(records, metrics, reference=None):
    from scipy.stats import spearmanr

    reference = Path(reference if reference is not None else PROJECT/'reference_results')
    with (reference/'01_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['sample_id', 'target_MOS', 'predicted_MOS']:
            raise ValueError('Supplied reference prediction columns differ.')
        expected_rows = list(reader)
    expected = {row['sample_id']: row for row in expected_rows}
    observed = {row['sample_id']: row for row in records}
    if (not expected or len(expected) != len(expected_rows) or len(observed) != len(records)
            or set(expected) != set(observed)):
        raise ValueError('Generated and supplied reference video identities differ.')
    names = list(expected)
    expected_values = np.asarray([[float(expected[name][key]) for key in ('target_MOS', 'predicted_MOS')]
                                  for name in names], dtype=np.float64)
    observed_values = np.asarray([[float(observed[name][key]) for key in ('target_MOS', 'predicted_MOS')]
                                  for name in names], dtype=np.float64)
    if not np.isfinite(expected_values).all() or not np.isfinite(observed_values).all():
        raise ValueError('Generated and reference scores must be finite.')
    if not np.array_equal(expected_values[:, 0], observed_values[:, 0]):
        raise ValueError('Generated and reference human MOS labels differ.')
    expected_metrics = json.loads((reference/'02_metrics.json').read_text('utf-8'))
    if metrics['videos'] != len(records) or expected_metrics['videos'] != len(records):
        raise ValueError('Generated and reference video counts differ.')
    expected_srcc = float(expected_metrics['SRCC'])
    observed_srcc = float(metrics['SRCC'])
    for values, score in ((expected_values, expected_srcc), (observed_values, observed_srcc)):
        computed = float(spearmanr(values[:, 0], values[:, 1]).statistic)
        if not np.isfinite(score) or not np.isclose(computed, score, rtol=0, atol=1e-12):
            raise ValueError('A reported SRCC differs from its per-video predictions.')
    maximum_mos_difference = float(np.max(np.abs(observed_values[:, 1] - expected_values[:, 1])))
    srcc_difference = abs(observed_srcc - expected_srcc)
    passed = maximum_mos_difference <= 1e-3 and srcc_difference <= 1e-4
    print('Reference comparison: ' + ('PASS' if passed else 'FAIL'), flush=True)
    print(f'Compared {len(records)} videos with {reference / "01_predictions.csv"} '
          f'and {reference / "02_metrics.json"}.', flush=True)
    print(f'Maximum MOS difference: {maximum_mos_difference:.6g} (tolerance 0.001); '
          f'SRCC difference: {srcc_difference:.6g} (tolerance 0.0001).', flush=True)
    if not passed:
        raise RuntimeError('Generated predictions or SRCC exceed the reference comparison tolerances.')
    return {'videos': len(records), 'maximum_MOS_difference': maximum_mos_difference,
            'SRCC_difference': srcc_difference, 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'weights/LightGen2.pt')
    parser.add_argument('--input', type=Path, default=PROJECT / 'data/input_for_review.npz',
                        help='LGVQ video frames')
    parser.add_argument('--reference-scores', type=Path, default=PROJECT / 'data/reference_mos.csv',
                        help='Human MOS labels used to calculate SRCC')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results', help='Result directory')
    parser.add_argument('--visualize-field', type=int, default=1,
                        help='One-based field number captured during the full inference')
    args = parser.parse_args()
    args.output = check_output_directory(args.output)
    torch.set_num_threads(4)
    if __package__:
        from .visualize import configure_figures
    else:
        from visualize import configure_figures
    configure_figures()
    model, _ = load_lightgen2(args.weights, args.device)
    with np.load(args.input, allow_pickle=False) as archive:
        sample_ids = read_input_metadata(archive)
        targets = read_reference_scores(args.reference_scores, sample_ids)
        records, metrics = evaluate(model, archive, sample_ids, targets, args.device, args.output,
                              visualize_field=args.visualize_field)
    metrics_path = args.output / '02_metrics.json'
    temporary = metrics_path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(metrics_path)
    print(f"SRCC {metrics['SRCC']} over {metrics['videos']} distinct videos")
    print('Saved predictions and optical fields to ' + str(args.output.resolve()))
    compare_reference_results(records, metrics)


if __name__ == '__main__':
    main()
