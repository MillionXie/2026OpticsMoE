"""Create plotting-ready raw and three-seed summary tables for the fixed-478 study."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SCALARS = [
    'val_accuracy', 'val_balanced_accuracy', 'val_macro_f1', 'val_macro_nll',
    'val_capture_mean', 'train_accuracy', 'train_balanced_accuracy',
    'train_macro_f1', 'train_macro_nll', 'train_capture_mean',
    'generalization_gap', 'best_epoch', 'elapsed_seconds', 'parameter_count',
]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def metric(payload, split, name):
    value = payload.get(split, {}).get(name)
    return float(value) if value is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    jobs_doc = json.loads((args.out / 'jobs.json').read_text())
    jobs = jobs_doc['moe'] + jobs_doc['baseline']
    rows, routes = [], []
    for job in jobs:
        folder = args.out / job['name']
        result_path = folder / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        metadata_path = folder / 'metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        row = {
            'dataset': job['dataset'], 'architecture': job['arch'],
            'experts': job['experts'], 'top_k': job['top_k'] if job['arch'] == 'moe_oeo' else '',
            'seed': job['seed'], 'run_name': job['name'],
            'val_accuracy': metric(result, 'val', 'accuracy'),
            'val_balanced_accuracy': metric(result, 'val', 'balanced_accuracy'),
            'val_macro_f1': metric(result, 'val', 'macro_f1'),
            'val_macro_nll': metric(result, 'val', 'macro_nll'),
            'val_capture_mean': metric(result, 'val', 'capture_mean'),
            'train_accuracy': metric(result, 'train', 'accuracy'),
            'train_balanced_accuracy': metric(result, 'train', 'balanced_accuracy'),
            'train_macro_f1': metric(result, 'train', 'macro_f1'),
            'train_macro_nll': metric(result, 'train', 'macro_nll'),
            'train_capture_mean': metric(result, 'train', 'capture_mean'),
            'best_epoch': result.get('best_epoch'),
            'elapsed_seconds': result.get('elapsed_seconds'),
            'parameter_count': metadata.get('parameter_count'),
            'checkpoint_sha256': result.get('checkpoint_sha256'),
            'test_read': result.get('test_read', False),
        }
        if row['train_accuracy'] is not None and row['val_accuracy'] is not None:
            row['generalization_gap'] = row['train_accuracy'] - row['val_accuracy']
        rows.append(row)
        for split in ['train', 'val']:
            values = result.get(split, {})
            probability = values.get('route_probability', [])
            load = values.get('route_load', [])
            for expert in range(max(len(probability), len(load))):
                routes.append({
                    'dataset': job['dataset'], 'architecture': job['arch'],
                    'experts': job['experts'], 'top_k': job['top_k'], 'seed': job['seed'],
                    'split': split, 'expert_index': expert,
                    'route_probability': probability[expert] if expert < len(probability) else '',
                    'route_load': load[expert] if expert < len(load) else '',
                    'distinct_selected_sets': values.get('distinct_selected_sets', ''),
                })

    raw_fields = ['dataset', 'architecture', 'experts', 'top_k', 'seed', 'run_name'] + SCALARS + [
        'checkpoint_sha256', 'test_read']
    write_csv(args.out / 'tables' / 'all_runs.csv', rows, raw_fields)
    route_fields = ['dataset', 'architecture', 'experts', 'top_k', 'seed', 'split',
                    'expert_index', 'route_probability', 'route_load', 'distinct_selected_sets']
    write_csv(args.out / 'tables' / 'routing_by_expert.csv', routes, route_fields)

    grouped = {}
    for row in rows:
        key = (row['dataset'], row['architecture'], row['experts'], row['top_k'])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, members in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        summary = dict(dataset=key[0], architecture=key[1], experts=key[2], top_k=key[3],
                       n_seeds=len(members), seeds=';'.join(str(x['seed']) for x in sorted(members, key=lambda x:x['seed'])))
        for name in SCALARS:
            values = [float(x[name]) for x in members if x.get(name) is not None and math.isfinite(float(x[name]))]
            summary[name + '_mean'] = statistics.fmean(values) if values else ''
            summary[name + '_std'] = statistics.stdev(values) if len(values) >= 2 else ''
        summaries.append(summary)
    summary_fields = ['dataset', 'architecture', 'experts', 'top_k', 'n_seeds', 'seeds'] + [
        suffix for name in SCALARS for suffix in (name + '_mean', name + '_std')]
    write_csv(args.out / 'tables' / 'summary_mean_std.csv', summaries, summary_fields)
    bundle = {
        'standard_deviation': 'sample standard deviation across random seeds (ddof=1)',
        'expected_seeds': [17, 27, 37], 'completed_runs': len(rows),
        'raw_runs': rows, 'summary': summaries, 'routing_by_expert': routes,
    }
    (args.out / 'tables' / 'plotting_bundle.json').write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'completed_runs': len(rows), 'groups': len(summaries)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
