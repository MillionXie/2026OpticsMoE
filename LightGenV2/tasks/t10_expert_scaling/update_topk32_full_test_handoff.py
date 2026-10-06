"""Build complete held-out-test trend tables for every trained Kather configuration."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_nll', 'capture_mean')
SELECTED_ON_VALIDATION = {4: 3, 16: 16, 25: 12, 49: 24}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None


def entropy(values):
    return -sum(x * math.log(x) for x in values if x > 0) / math.log(len(values)) if len(values) > 1 else 1.0


def cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    args = ap.parse_args()
    package = args.package
    evidence = package / 'evidence'
    lock_path = package / 'full_test_scan_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    manifest = json.loads((evidence / 'dataset' / 'data_manifest.json').read_text(encoding='utf-8'))
    training = read_csv(package / 'per_seed_runs.csv')
    training_by_name = {x['run_name']: x for x in training}

    rows, confusions, routes = [], [], []
    for spec in lock['runs']:
        name = spec['run_name']
        result = json.loads((evidence / 'runs' / name / 'test_result.json').read_text(encoding='utf-8'))
        if result['state'] != 'complete' or result['selection_lock_sha256'] != lock_hash:
            raise AssertionError(f'Invalid full-test record: {name}')
        if result['checkpoint_sha256'] != spec['checkpoint_sha256'] or result['test_samples'] != 752:
            raise AssertionError(f'Provenance mismatch: {name}')
        train = training_by_name[name]
        row = {
            'run_name': name, 'architecture': spec['architecture'], 'experts': spec['experts'],
            'top_k': spec['top_k'], 'top_k_fraction': spec['top_k'] / spec['experts'],
            'seed': spec['seed'], 'selected_on_validation': (
                spec['architecture'] == 'moe_oeo' and spec['top_k'] == SELECTED_ON_VALIDATION[spec['experts']]),
            'val_accuracy': float(train['val_accuracy']), 'val_macro_f1': float(train['val_macro_f1']),
            'test_samples': result['test_samples'], 'checkpoint_sha256': result['checkpoint_sha256'],
            'full_test_scan_lock_sha256': lock_hash, 'data_sha256': result['data_sha256'],
            'test_ids_sha256': result['test_ids_sha256'],
        }
        for metric in METRICS:
            row['test_' + metric] = result['test'][metric]
        if 'route_probability' in result['test']:
            prob, load = result['test']['route_probability'], result['test']['route_load']
            row['test_route_entropy_normalized'] = entropy(prob)
            row['test_route_load_cv'] = cv(load)
            row['test_distinct_selected_sets'] = result['test']['distinct_selected_sets']
            for i, (p, l) in enumerate(zip(prob, load)):
                routes.append({'run_name': name, 'experts': spec['experts'], 'top_k': spec['top_k'],
                               'top_k_fraction': spec['top_k']/spec['experts'], 'seed': spec['seed'],
                               'expert_index': i, 'soft_probability_mean': p,
                               'hard_selection_fraction': l,
                               'distinct_selected_sets': result['test']['distinct_selected_sets']})
        rows.append(row)
        for ti, cm_row in enumerate(result['test']['confusion_matrix']):
            for pi, count in enumerate(cm_row):
                confusions.append({'run_name': name, 'architecture': spec['architecture'],
                                   'experts': spec['experts'], 'top_k': spec['top_k'], 'seed': spec['seed'],
                                   'true_class_index': ti, 'true_class_name': manifest['classes'][ti],
                                   'predicted_class_index': pi, 'predicted_class_name': manifest['classes'][pi],
                                   'count': count})
    if len(rows) != 40:
        raise AssertionError(len(rows))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['architecture'], int(row['experts']), int(row['top_k']))].append(row)
    summary = []
    for (arch, n, k), members in sorted(grouped.items()):
        if sorted(int(x['seed']) for x in members) != [17, 27]:
            raise AssertionError((arch, n, k))
        item = {'architecture': arch, 'experts': n, 'top_k': k, 'top_k_fraction': k/n,
                'selected_on_validation': arch == 'moe_oeo' and k == SELECTED_ON_VALIDATION[n],
                'n_seeds': 2, 'seeds': '17;27'}
        for metric in METRICS:
            item[f'test_{metric}_mean'], item[f'test_{metric}_sd'] = mean_sd(
                [float(x[f'test_{metric}']) for x in members])
        item['val_accuracy_mean'], item['val_accuracy_sd'] = mean_sd([float(x['val_accuracy']) for x in members])
        if arch == 'moe_oeo':
            for metric in ('test_route_entropy_normalized', 'test_route_load_cv', 'test_distinct_selected_sets'):
                item[metric + '_mean'], item[metric + '_sd'] = mean_sd([float(x[metric]) for x in members])
        summary.append(item)

    trends = []
    for moe in (x for x in summary if x['architecture'] == 'moe_oeo'):
        d2nn = next(x for x in summary if x['architecture'] == 'd2nn_expert_global' and x['experts'] == moe['experts'])
        trends.append({
            'experts': moe['experts'], 'top_k': moe['top_k'], 'top_k_fraction': moe['top_k_fraction'],
            'selected_on_validation': moe['selected_on_validation'],
            'moe_test_accuracy_mean': moe['test_accuracy_mean'], 'moe_test_accuracy_sd': moe['test_accuracy_sd'],
            'd2nn_test_accuracy_mean': d2nn['test_accuracy_mean'], 'd2nn_test_accuracy_sd': d2nn['test_accuracy_sd'],
            'moe_minus_d2nn_test_pp': 100*(moe['test_accuracy_mean']-d2nn['test_accuracy_mean']),
            'moe_test_macro_f1_mean': moe['test_macro_f1_mean'], 'moe_test_macro_f1_sd': moe['test_macro_f1_sd'],
            'd2nn_test_macro_f1_mean': d2nn['test_macro_f1_mean'],
            'moe_test_route_entropy_mean': moe['test_route_entropy_normalized_mean'],
            'moe_test_route_load_cv_mean': moe['test_route_load_cv_mean'],
            'moe_test_distinct_selected_sets_mean': moe['test_distinct_selected_sets_mean'],
        })
    trends.sort(key=lambda x: (x['experts'], x['top_k']))

    full_by_name = {x['run_name']: x for x in rows}
    appended = ['test_evaluated'] + [f'test_{x}' for x in METRICS]
    for row in training:
        full = full_by_name[row['run_name']]
        row['test_evaluated'] = True
        for key in appended[1:]:
            row[key] = full[key]
    write_csv(package / 'per_seed_runs.csv', training, list(training[0]))
    write_csv(package / 'full_test_per_seed.csv', rows)
    fields = sorted({k for x in summary for k in x}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 999)
    write_csv(package / 'full_test_summary.csv', summary, fields)
    write_csv(package / 'topk_test_trends.csv', trends)
    write_csv(package / 'full_test_confusion_matrices.csv', confusions)
    write_csv(package / 'full_test_expert_routes.csv', routes)

    plotting_path = package / 'plotting_summary.json'
    plotting = json.loads(plotting_path.read_text(encoding='utf-8'))
    plotting['completion']['test_evaluated_runs'] = 40
    plotting['completion']['test_samples_per_run'] = 752
    plotting['completion']['full_test_scan_policy'] = lock['test_policy']
    plotting['full_test_summary'] = summary
    plotting['topk_test_trends'] = trends
    plotting_path.write_text(json.dumps(plotting, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = ['# Kather2016 固定专家尺寸、专家数与 Top-k 完整测试消融', '',
             '本包包含 32 组 MoE 和 8 组参数匹配 D2NN 的完整测试集推理。每组使用固定的最佳验证 macro-NLL checkpoint，在同一 752 张测试图像上推理。', '',
             '## 完整测试趋势', '',
             '|专家数 N|top-k|k/N|MoE 测试准确率（均值±SD）|D2NN 测试准确率（均值±SD）|差值（百分点）|验证集原选定|',
             '|---:|---:|---:|---:|---:|---:|:---:|']
    for x in trends:
        lines.append(f"|{x['experts']}|{x['top_k']}|{x['top_k_fraction']:.3f}|{x['moe_test_accuracy_mean']*100:.2f}±{x['moe_test_accuracy_sd']*100:.2f}|{x['d2nn_test_accuracy_mean']*100:.2f}±{x['d2nn_test_accuracy_sd']*100:.2f}|{x['moe_minus_d2nn_test_pp']:.2f}|{'是' if x['selected_on_validation'] else ''}|")
    lines += ['', '## 使用说明', '',
              '- `topk_test_trends.csv`：作图首选，包含全部 16 个 MoE `(N,k)` 组合及相同 N 的 D2NN 参考线。',
              '- `full_test_summary.csv`：20 个架构配置的测试均值和样本标准差。',
              '- `full_test_per_seed.csv`：40 个 checkpoint 的逐种子验证/测试指标与哈希。',
              '- `full_test_expert_routes.csv`：全部 MoE 配置的测试集 soft 概率和 hard 选中比例。',
              '- `full_test_confusion_matrices.csv`：全部配置的测试集混淆矩阵长表。',
              '- `full_test_scan_lock.json`：测试前冻结的完整运行列表和 checkpoint 哈希。',
              '- `test_selection_lock.json` 与 `locked_test_*`：此前严格按验证集选 k 后得到的 16 组结果，数值包含于完整扫描中，保留用于溯源。', '',
              '完整测试网格适合分析趋势。若据此重新选择 top-k，则所选配置的数值属于探索性测试结果；正式最终报告应固定选择规则或另设最终测试集。', '',
              '## 固定实验设置', '',
              '- Kather2016：8 类、5000 张 150×150 RGB 图像，CC BY 4.0；固定划分 train 3496、validation 752、test 752。',
              '- 单专家 224×224，间隔 30 像素；专家数 4、16、25、49。逻辑输入采样间距 17 μm，相位器件映射间距 8 μm。',
              '- 4 个主干相位层且逐层 OEO；MoE 使用 2 个专家相位层和 2 个 global 相位层，D2NN 使用 4 个连续相位层。',
              '- 动态路由为 hard top-k 前向与 dense amplitude straight-through 反向，soft 路由均衡损失权重 0.01。',
              '- 训练 60 个 epoch，AdamW，学习率 0.002，effective batch 16，microbatch 2，EMA 0.95。']
    (package / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.iterdir())
              if p.is_file() and p.name != 'SHA256.json'}
    (package / 'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'test_runs': len(rows), 'summary_groups': len(summary), 'trends': trends}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
