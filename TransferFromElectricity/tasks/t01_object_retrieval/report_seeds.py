"""Compare paired optimization seeds without mixing datasets or update budgets."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(runs, output):
    pairs, contracts, evidence = {}, set(), []
    for run in runs:
        if read(run / 'status.json')['status'] != 'complete':
            raise ValueError(f'Incomplete run: {run}')
        cfg = read(run / 'protocol.json')
        result = read(run / 'final_report.json')
        env = read(run / 'environment.json')
        method, seed = result['method'], cfg['seed']
        if method not in {'direct', 'qwen_lora'} or method != cfg['method']:
            raise ValueError('Only paired direct/Qwen-LoRA runs are supported')
        if method in pairs.setdefault(seed, {}):
            raise ValueError(f'Duplicate seed/method: {seed}/{method}')
        config = {k: v for k, v in cfg.items() if k not in {'seed', 'method'}}
        contracts.add((result['git_sha'], result['split_sha256'], env['device'],
                       result['optimizer_updates'], json.dumps(config, sort_keys=True)))
        pairs[seed][method] = {
            'run_id': run.name, 'selected_epoch': result['selected_epoch'],
            'top1': result['selected_live_test']['top1_retrieval_accuracy'],
            'phase_rms_rad': result['selected_expert_phase']['rms_change_rad'],
            'initial_experts_top1': result['ablations']['initial_experts']['top1_retrieval_accuracy'],
            'lora_disabled_top1': result['ablations'].get('lora_disabled_same_decoder', {}).get('top1_retrieval_accuracy'),
            'export_max_error': result['export_max_error']}
        for name in ('final_report.json', 'protocol.json', 'environment.json', 'transfer_manifest.json'):
            path = run / name
            if path.exists():
                evidence.append({'run_id': run.name, 'file': name,
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(contracts) != 1 or len(pairs) < 2 or any(set(p) != {'direct', 'qwen_lora'} for p in pairs.values()):
        raise ValueError('Require complete seed pairs sharing source, split, device model, budget and configuration')
    sha, split, device, updates, _ = next(iter(contracts))
    rows = [{'seed': s, **p, 'qwen_minus_direct_pp': 100 * (p['qwen_lora']['top1'] - p['direct']['top1'])}
            for s, p in sorted(pairs.items())]
    report = {'training_git_sha': sha, 'split_sha256': split, 'device': device,
              'updates_per_run': updates, 'rows': rows,
              'summary': {m: {'mean_top1': statistics.mean(p[m]['top1'] for p in pairs.values()),
                              'sample_std_top1': statistics.stdev(p[m]['top1'] for p in pairs.values())}
                          for m in ('direct', 'qwen_lora')},
              'limitations': ['Same test queries reused across seeds, not independent new test samples',
                              'Optimization stability only; no claim of statistical significance',
                              'Historical Caltech warmstart and prior test inspection remain limitations',
                              'Ideal optical simulation; no hardware robustness conclusion']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'evidence_manifest.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for method, label, color in [('direct', 'Direct phase', '#1976b5'), ('qwen_lora', 'Qwen LoRA + decoder', '#ad3375')]:
        axes[0].plot([r['seed'] for r in rows], [100*r[method]['top1'] for r in rows], 'o-', label=label, color=color)
        axes[1].plot([r['seed'] for r in rows], [r[method]['phase_rms_rad'] for r in rows], 'o-', color=color)
    for ax in axes:
        ax.set_xlabel('Optimization seed')
        ax.set_xticks(sorted(pairs))
    axes[0].set_ylabel('Selected test Top-1 (%)')
    axes[0].set_title(f'Paired seeds: {updates:,} updates each')
    axes[0].legend()
    axes[1].set_ylabel('Circular phase RMS change (rad)')
    axes[1].set_title('Selected expert movement from initialization')
    fig.savefig(output / 'paired_seeds.png', dpi=170)
    plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize([Path(p) for p in args.runs], Path(args.output))
