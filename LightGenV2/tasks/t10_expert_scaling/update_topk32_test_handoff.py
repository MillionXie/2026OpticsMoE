"""Add locked held-out test results to the Kather expert-count handoff."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


TEST_METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_nll', 'capture_mean')


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


def normalized_entropy(values):
    if len(values) <= 1:
        return 1.0
    return -sum(x * math.log(x) for x in values if x > 0) / math.log(len(values))


def coefficient_of_variation(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    args = ap.parse_args()
    package = args.package
    evidence = package / 'evidence'
    lock = json.loads((package / 'test_selection_lock.json').read_text(encoding='utf-8'))
    lock_hash = hashlib.sha256((package / 'test_selection_lock.json').read_bytes()).hexdigest()
    manifest = json.loads((evidence / 'dataset' / 'data_manifest.json').read_text(encoding='utf-8'))
    per_seed = read_csv(package / 'per_seed_runs.csv')
    by_name = {r['run_name']: r for r in per_seed}

    locked_rows, confusions, routes = [], [], []
    for spec in lock['runs']:
        name = spec['run_name']
        result = json.loads((evidence / 'runs' / name / 'test_result.json').read_text(encoding='utf-8'))
        if result['state'] != 'complete' or result['selection_lock_sha256'] != lock_hash:
            raise AssertionError(f'Invalid test record: {name}')
        if result['checkpoint_sha256'] != spec['checkpoint_sha256'] or result['test_samples'] != 752:
            raise AssertionError(f'Test provenance mismatch: {name}')
        base = by_name[name]
        row = {
            'run_name': name, 'architecture': spec['architecture'], 'experts': spec['experts'],
            'top_k': spec['top_k'], 'top_k_fraction': spec['top_k'] / spec['experts'],
            'seed': spec['seed'], 'selection_split': 'validation',
            'selection_rule': 'maximum mean validation accuracy across seeds 17 and 27 within each expert count',
            'checkpoint_selection': 'minimum validation macro-NLL within the training run',
            'val_accuracy': float(base['val_accuracy']), 'val_macro_f1': float(base['val_macro_f1']),
            'test_samples': result['test_samples'], 'checkpoint_sha256': result['checkpoint_sha256'],
            'selection_lock_sha256': result['selection_lock_sha256'],
            'data_sha256': result['data_sha256'], 'test_ids_sha256': result['test_ids_sha256'],
        }
        for metric in TEST_METRICS:
            row['test_' + metric] = result['test'][metric]
        if 'route_probability' in result['test']:
            prob = result['test']['route_probability']
            load = result['test']['route_load']
            row['test_route_entropy_normalized'] = normalized_entropy(prob)
            row['test_route_load_cv'] = coefficient_of_variation(load)
            row['test_distinct_selected_sets'] = result['test']['distinct_selected_sets']
            for i, (p, l) in enumerate(zip(prob, load)):
                routes.append({
                    'run_name': name, 'experts': spec['experts'], 'top_k': spec['top_k'],
                    'seed': spec['seed'], 'expert_index': i,
                    'soft_probability_mean': p, 'hard_selection_fraction': l,
                    'distinct_selected_sets': result['test']['distinct_selected_sets'],
                })
        locked_rows.append(row)
        for true_index, cm_row in enumerate(result['test']['confusion_matrix']):
            for pred_index, count in enumerate(cm_row):
                confusions.append({
                    'run_name': name, 'architecture': spec['architecture'],
                    'experts': spec['experts'], 'top_k': spec['top_k'], 'seed': spec['seed'],
                    'true_class_index': true_index, 'true_class_name': manifest['classes'][true_index],
                    'predicted_class_index': pred_index, 'predicted_class_name': manifest['classes'][pred_index],
                    'count': count,
                })

    if len(locked_rows) != 16:
        raise AssertionError(len(locked_rows))
    grouped = defaultdict(list)
    for row in locked_rows:
        grouped[(row['architecture'], int(row['experts']), int(row['top_k']))].append(row)
    summary = []
    for (arch, n, k), members in sorted(grouped.items()):
        if sorted(int(x['seed']) for x in members) != [17, 27]:
            raise AssertionError((arch, n, k))
        item = {'architecture': arch, 'experts': n, 'top_k': k,
                'top_k_fraction': k / n, 'n_seeds': 2, 'seeds': '17;27'}
        for metric in TEST_METRICS:
            item[f'test_{metric}_mean'], item[f'test_{metric}_sd'] = mean_sd(
                [float(x[f'test_{metric}']) for x in members])
        item['val_accuracy_mean'], item['val_accuracy_sd'] = mean_sd([float(x['val_accuracy']) for x in members])
        if arch == 'moe_oeo':
            for metric in ('test_route_entropy_normalized', 'test_route_load_cv', 'test_distinct_selected_sets'):
                item[metric + '_mean'], item[metric + '_sd'] = mean_sd([float(x[metric]) for x in members])
        summary.append(item)

    comparisons, paired = [], []
    for n in (4, 16, 25, 49):
        moe = next(x for x in summary if x['architecture'] == 'moe_oeo' and x['experts'] == n)
        d2nn = next(x for x in summary if x['architecture'] == 'd2nn_expert_global' and x['experts'] == n)
        comparisons.append({
            'experts': n, 'selected_top_k': moe['top_k'],
            'selection_rule': 'top-k selected by mean validation accuracy; test evaluated once after locking',
            'moe_test_accuracy_mean': moe['test_accuracy_mean'], 'moe_test_accuracy_sd': moe['test_accuracy_sd'],
            'd2nn_test_accuracy_mean': d2nn['test_accuracy_mean'], 'd2nn_test_accuracy_sd': d2nn['test_accuracy_sd'],
            'moe_minus_d2nn_test_pp': 100 * (moe['test_accuracy_mean'] - d2nn['test_accuracy_mean']),
            'moe_test_macro_f1_mean': moe['test_macro_f1_mean'],
            'd2nn_test_macro_f1_mean': d2nn['test_macro_f1_mean'],
            'moe_val_accuracy_mean': moe['val_accuracy_mean'], 'd2nn_val_accuracy_mean': d2nn['val_accuracy_mean'],
        })
        for seed in (17, 27):
            m = next(x for x in locked_rows if x['architecture'] == 'moe_oeo' and int(x['experts']) == n and int(x['seed']) == seed)
            d = next(x for x in locked_rows if x['architecture'] == 'd2nn_expert_global' and int(x['experts']) == n and int(x['seed']) == seed)
            paired.append({'experts': n, 'selected_top_k': moe['top_k'], 'seed': seed,
                           'moe_test_accuracy': m['test_accuracy'], 'd2nn_test_accuracy': d['test_accuracy'],
                           'moe_minus_d2nn_test_pp': 100 * (m['test_accuracy'] - d['test_accuracy'])})

    # Keep the 40-run validation scan intact and append held-out metrics only to locked rows.
    locked_by_name = {x['run_name']: x for x in locked_rows}
    appended = ['test_evaluated'] + [f'test_{m}' for m in TEST_METRICS]
    for row in per_seed:
        test = locked_by_name.get(row['run_name'])
        row['test_evaluated'] = bool(test)
        for key in appended[1:]:
            row[key] = test[key] if test else ''
    write_csv(package / 'per_seed_runs.csv', per_seed, list(per_seed[0]))
    write_csv(package / 'locked_test_per_seed.csv', locked_rows)
    write_csv(package / 'locked_test_summary.csv', summary, sorted({k for x in summary for k in x}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 999))
    write_csv(package / 'best_topk_vs_d2nn.csv', comparisons)
    write_csv(package / 'paired_seed_differences.csv', paired)
    write_csv(package / 'test_confusion_matrices.csv', confusions)
    write_csv(package / 'test_expert_routes.csv', routes)

    plotting_path = package / 'plotting_summary.json'
    plotting = json.loads(plotting_path.read_text(encoding='utf-8'))
    plotting['completion']['test_evaluated_runs'] = 16
    plotting['completion']['test_samples_per_run'] = 752
    plotting['completion']['test_policy'] = lock['test_policy']
    plotting['locked_test_summary'] = summary
    plotting['best_topk_vs_d2nn'] = comparisons
    plotting['paired_seed_differences'] = paired
    plotting_path.write_text(json.dumps(plotting, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Kather2016 固定专家尺寸、专家数与 Top-k 消融', '',
        '本包包含 32 组 MoE 扫描、8 组参数匹配 D2NN 对照，以及锁定配置后的 16 次独立测试集推理。', '',
        '## 结果使用规则', '',
        '- 验证集仅用于两个选择：每个专家数选择平均验证准确率最高的 top-k；每次训练内部按最低验证 macro-NLL 保存 checkpoint。',
        '- 完成选择后写入 `test_selection_lock.json`，随后每个锁定 checkpoint 只在测试集推理一次。主结果使用测试集，不用验证集代替测试集。',
        '- 测试集共 752 张图像；两个随机种子为 17 和 27。均值和标准差均跨随机种子计算，标准差为样本标准差（ddof=1）。', '',
        '## 测试集主结果', '',
        '|专家数 N|验证集选定 top-k|MoE 测试准确率（均值±SD）|D2NN 测试准确率（均值±SD）|MoE-D2NN（百分点）|',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in comparisons:
        lines.append(f"|{row['experts']}|{row['selected_top_k']}|{row['moe_test_accuracy_mean']*100:.2f}±{row['moe_test_accuracy_sd']*100:.2f}|{row['d2nn_test_accuracy_mean']*100:.2f}±{row['d2nn_test_accuracy_sd']*100:.2f}|{row['moe_minus_d2nn_test_pp']:.2f}|")
    lines += ['', '## 数据与模型设置', '',
        '- Kather2016：8 类、5000 张 150×150 RGB 组织图像，许可为 CC BY 4.0。固定图像级划分为 train 3496、validation 752、test 752；三个集合的样本 ID 无交集。原数据不提供患者 ID，因此无法验证患者级独立性。',
        '- 单专家尺寸固定为 224×224，相邻专家间隔 30 像素；专家数为 4、16、25、49，对应 2×2、4×4、5×5、7×7。逻辑输入采样间距 17 μm，相位器件映射间距 8 μm。',
        '- RGB 三通道分别插值后拼成 `[R,G;B,mean]` 单幅振幅图。MoE 将 224×224 编码输入复制到各专家；D2NN 将原图直接编码至其连续相位孔径的完整尺寸，使全孔径受到输入调制。',
        '- 每个模型含 4 个主干相位层且每层后接 OEO。MoE 交替使用 2 个专家相位层和 2 个 global 相位层；D2NN 使用 4 个连续相位层。没有电子残差、CNN 或 Qwen 分类支路。',
        '- 动态路由前向采用 hard top-k，反向采用 dense amplitude straight-through 近似。均衡损失约束 microbatch 内平均 soft 路由概率，权重 0.01。',
        '- D2NN 匹配 MoE 的专家相位参数与 global 相位参数之和，不计路由器相位参数；精确参数及取整误差见 `geometry_and_parameters.csv`。',
        '- 训练 60 个 epoch，AdamW，学习率 0.002，effective batch 16，microbatch 2，EMA 0.95。', '',
        '## 文件说明', '',
        '- `locked_test_per_seed.csv`：16 个锁定 checkpoint 的逐种子验证与测试指标及溯源哈希。',
        '- `locked_test_summary.csv`：按架构、专家数和 top-k 汇总的测试均值与样本标准差。',
        '- `best_topk_vs_d2nn.csv`：用于主图的 MoE 与 D2NN 测试集比较。',
        '- `paired_seed_differences.csv`：同一专家数、同一随机种子下的测试准确率差。',
        '- `test_confusion_matrices.csv`：测试集混淆矩阵长表；行是真实类别，列是预测类别。',
        '- `test_expert_routes.csv`：测试集 soft 路由概率与 hard 选中比例。hard 比例跨专家求和为 k，soft 概率跨专家求和为 1。',
        '- `per_seed_runs.csv` 和 `summary_mean_sd.csv`：完整 40 组训练/验证扫描。未锁定配置的测试字段保持为空。',
        '- `learning_curves.csv`、`expert_routes.csv` 和 `confusion_matrices.csv`：训练过程与验证集诊断，仅用于选型和过拟合检查。',
        '- `evidence/runs/*/test_result.json` 与 `test_predictions.npz`：逐 checkpoint 测试指标、样本 ID、类别标签、类别分数和预测。', '',
        '## 作图建议', '',
        '主图使用 `best_topk_vs_d2nn.csv`：横轴为专家数 N，分别绘制 MoE 和 D2NN 的测试准确率均值，误差条为两个随机种子的样本标准差，并在图注注明 “top-k selected on validation”。完整 top-k 规律仍使用验证集扫描展示，不能标成测试集结果。',
    ]
    (package / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    hashes = {}
    for path in sorted(package.iterdir()):
        if path.is_file() and path.name != 'SHA256.json':
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (package / 'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'test_runs': len(locked_rows), 'summary': comparisons}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
