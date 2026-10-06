"""Summarize completed, matched pilots without copying checkpoints or datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', nargs=4, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, evidence, curves, contracts = [], [], [], []
    for arg in args.runs:
        run = Path(arg)
        if read(run/'status.json')['status'] != 'complete':
            raise ValueError(f'Incomplete run: {run}')
        final = read(run/'final_report.json')
        env = read(run/'environment.json')
        architecture = read(run/'architecture.json')
        gradients = read(run/'gradient_chain.json')
        config = read(run/'pilot_config.json')
        contracts.append((final['git_sha'], env['split_sha256'], config['epochs'], config['seed'], final['optimizer_steps']))
        with (run/'train_log.csv').open() as f:
            log = list(csv.DictReader(f))
        metrics = final['metrics']
        rows.append({'method': final['method'], 'run_id': run.name,
            'top1': metrics['top1_retrieval_accuracy'], 'top3': metrics['top3_retrieval_accuracy'], 'mrr': metrics['mrr'],
            'train_samples': final['train_samples'], 'optimizer_steps': final['optimizer_steps'],
            'peak_memory_gib': final['peak_memory_gib'], 'train_and_eval_seconds': final['elapsed_seconds'],
            'total_trainable': architecture['total_trainable'], 'generator_trainable': architecture['generator_trainable'],
            'initial_max_error': architecture['initial_max_error'],
            'first_task_loss': float(log[0]['task_loss']),
            'last_epoch_mean_task_loss': sum(float(x['task_loss']) for x in log if int(x['epoch']) == config['epochs']) / sum(int(x['epoch']) == config['epochs'] for x in log),
            'expert_phase_rms_change_rad': final['physical_phase_rms_change_rad'],
            'export_max_error': final['export_max_error'], 'batch_max_error': final['batch_max_error'],
            'order_max_error': final['order_max_error'], 'task_gradient_chain': gradients,
            'train_expert_selection_counts': final['train_expert_selection_counts']})
        curves.append((final['method'], log))
        for p in sorted(run.iterdir()):
            if p.suffix in {'.json','.yaml','.csv'}:
                evidence.append({'run_id': run.name, 'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size})
    if len(set(contracts)) != 1:
        raise ValueError(f'Incomparable pilot contracts: {contracts}')
    if {x['method'] for x in rows} != {'direct','small_hyper','qwen_frozen','qwen_lora'}:
        raise ValueError('Expected all four methods')
    result = {'protocol': '300 training images, 50 steps, final EMA, one seed; not full-data benchmark',
              'git_sha': contracts[0][0], 'split_sha256': contracts[0][1], 'seed': contracts[0][3],
              'test_used_for_checkpoint_selection': False, 'rows': rows,
              'time_boundary': 'training loop + checkpoint I/O + final evaluations; excludes model loading/source hashing; CUDA simulation, not optical latency'}
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    (output/'evidence_manifest.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout='constrained')
    labels = {'direct':'Direct mask', 'small_hyper':'Small generator', 'qwen_frozen':'Frozen Qwen + head', 'qwen_lora':'Qwen LoRA + head'}
    for method, log in curves:
        epochs = sorted({int(x['epoch']) for x in log})
        means = [sum(float(x['task_loss']) for x in log if int(x['epoch'])==e)/sum(int(x['epoch'])==e for x in log) for e in epochs]
        axes[0].plot(epochs, means, marker='o', label=labels[method])
    axes[0].set(xlabel='Epoch', ylabel='Mean training task loss', title='Same 300-image subset / 50 updates', xticks=[1,2,3,4,5])
    axes[0].legend(fontsize=8)
    bars = axes[1].bar([labels[x['method']] for x in rows], [100*x['top1'] for x in rows], color=['#64748b','#d97706','#0d9488','#4f46e5'])
    axes[1].bar_label(bars, fmt='%.1f%%')
    axes[1].set(ylabel='Test Top-1 (%)', ylim=(0,100), title='Final EMA / 200 test queries / seed 42')
    axes[1].tick_params(axis='x', labelrotation=20)
    fig.savefig(output/'pilot_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
