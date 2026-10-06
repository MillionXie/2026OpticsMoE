"""Build a plotting handoff for the completed Kather expert-count/top-k scan."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from .handoff_output import prepare_output


METRICS = [
    'train_accuracy', 'train_balanced_accuracy', 'train_macro_f1', 'train_macro_nll',
    'val_accuracy', 'val_balanced_accuracy', 'val_macro_f1', 'val_macro_nll',
    'val_capture_mean', 'generalization_gap', 'best_epoch', 'elapsed_seconds',
]


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None


def normalized_entropy(probabilities):
    n = len(probabilities)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy / math.log(n)


def coefficient_of_variation(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def router_cells(n, side, width=24, pitch=32):
    grid = math.isqrt(n)
    start = (side - ((grid - 1) * pitch + width)) // 2
    cells = [(y, x) for y in range(grid) for x in range(grid)]
    cells.sort(key=lambda p: ((2*p[0]-(grid-1))**2 + (2*p[1]-(grid-1))**2, p[0], p[1]))
    return [(y, x, start+y*pitch+width/2, start+x*pitch+width/2) for y, x in cells]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    evidence = args.package / 'evidence'
    package = prepare_output(args.package, args.output, [
        'evidence/selected_jobs.json', 'evidence/dataset/data_manifest.json',
        'evidence/scan/scan_design.json'])
    jobs = json.loads((evidence / 'selected_jobs.json').read_text())
    manifest = json.loads((evidence / 'dataset/data_manifest.json').read_text())
    scan_design = json.loads((evidence / 'scan/scan_design.json').read_text())

    raw, routes, curves, confusion = [], [], [], []
    commits, data_hashes = set(), set()
    geometry_by_arch_n = {}
    for job in jobs:
        folder = evidence / 'runs' / job['name']
        result = json.loads((folder / 'result.json').read_text())
        metadata = json.loads((folder / 'metadata.json').read_text())
        history = json.loads((folder / 'history.json').read_text())
        actual = metadata['arguments']
        if result.get('test_read') is not False or metadata.get('test_read') is not False:
            raise AssertionError(f"Test data were accessed: {job['name']}")
        for key in ['arch', 'experts', 'top_k', 'seed']:
            if actual[key] != job[key]:
                raise AssertionError((job['name'], key, actual[key], job[key]))
        commits.add(metadata['git_commit'])
        data_hashes.add(metadata['data_sha256'])
        geo = metadata['geometry']
        geometry_by_arch_n[(job['arch'], job['experts'])] = (geo, metadata['parameter_count'])
        row = dict(
            run_name=job['name'], architecture=job['arch'], experts=job['experts'],
            top_k=job['top_k'], top_k_fraction=job['top_k']/job['experts'], seed=job['seed'],
            layers=actual['layers'], learning_rate=actual['lr'], epochs=actual['epochs'],
            microbatch=actual['microbatch'], parameter_count=metadata['parameter_count'],
            best_epoch=result['best_epoch'], elapsed_seconds=result['elapsed_seconds'],
            checkpoint_sha256=result['checkpoint_sha256'], git_commit=metadata['git_commit'],
            data_sha256=metadata['data_sha256'], test_read=False,
        )
        for split in ['train', 'val']:
            values = result[split]
            for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_nll', 'capture_mean']:
                row[f'{split}_{metric}'] = values[metric]
            if job['arch'] == 'moe_oeo':
                prob, load = values['route_probability'], values['route_load']
                if abs(sum(prob)-1) > 1e-4 or abs(sum(load)-job['top_k']) > 1e-4:
                    raise AssertionError((job['name'], split, sum(prob), sum(load)))
                row[f'{split}_route_entropy_normalized'] = normalized_entropy(prob)
                row[f'{split}_route_load_cv'] = coefficient_of_variation(load)
                row[f'{split}_distinct_selected_sets'] = values['distinct_selected_sets']
                ports = router_cells(job['experts'], geo['router_side_px'])
                grid = math.isqrt(job['experts'])
                for index, (p, l) in enumerate(zip(prob, load)):
                    port_row, port_col, port_y, port_x = ports[index]
                    routes.append(dict(
                        run_name=job['name'], experts=job['experts'], top_k=job['top_k'],
                        top_k_fraction=job['top_k']/job['experts'], seed=job['seed'], split=split,
                        expert_index=index, expert_grid_row=index//grid, expert_grid_col=index%grid,
                        router_port_row=port_row, router_port_col=port_col,
                        router_port_center_y_px=port_y, router_port_center_x_px=port_x,
                        soft_probability_mean=p, hard_selection_fraction=l,
                        distinct_selected_sets=values['distinct_selected_sets']))
            for true_class, cm_row in enumerate(values['confusion_matrix']):
                for predicted_class, count in enumerate(cm_row):
                    confusion.append(dict(
                        run_name=job['name'], architecture=job['arch'], experts=job['experts'],
                        top_k=job['top_k'], seed=job['seed'], split=split,
                        true_class_index=true_class, true_class_name=manifest['classes'][true_class],
                        predicted_class_index=predicted_class,
                        predicted_class_name=manifest['classes'][predicted_class], count=count))
        row['generalization_gap'] = row['train_accuracy'] - row['val_accuracy']
        raw.append(row)
        for entry in history:
            val = entry['val']
            curve = dict(
                run_name=job['name'], architecture=job['arch'], experts=job['experts'],
                top_k=job['top_k'], top_k_fraction=job['top_k']/job['experts'], seed=job['seed'],
                epoch=entry['epoch'], train_objective=entry['train_objective'],
                learning_rate=entry['lr'], elapsed_seconds=entry['elapsed_seconds'],
                val_accuracy=val['accuracy'], val_balanced_accuracy=val['balanced_accuracy'],
                val_macro_f1=val['macro_f1'], val_macro_nll=val['macro_nll'],
                val_capture_mean=val['capture_mean'])
            if 'route_probability' in val:
                curve['val_route_entropy_normalized'] = normalized_entropy(val['route_probability'])
                curve['val_route_load_cv'] = coefficient_of_variation(val['route_load'])
                curve['val_distinct_selected_sets'] = val['distinct_selected_sets']
            curves.append(curve)

    if len(raw) != 40 or sum(r['architecture'] == 'moe_oeo' for r in raw) != 32:
        raise AssertionError(len(raw))
    if len(commits) != 1 or len(data_hashes) != 1:
        raise AssertionError((commits, data_hashes))
    if next(iter(data_hashes)) != manifest['cache_sha256']:
        raise AssertionError('Dataset cache hash differs from manifest')

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row['architecture'], row['experts'], row['top_k'])].append(row)
    summary = []
    for (arch, n, k), members in sorted(grouped.items()):
        if sorted(r['seed'] for r in members) != [17, 27]:
            raise AssertionError((arch, n, k))
        item = dict(architecture=arch, experts=n, top_k=k, top_k_fraction=k/n,
                    n_seeds=len(members), seeds='17;27')
        for metric in METRICS:
            values = [float(r[metric]) for r in members]
            item[metric+'_mean'], item[metric+'_sd'] = mean_sd(values)
        summary.append(item)

    comparisons, paired = [], []
    for n in [4, 16, 25, 49]:
        moe_groups = [x for x in summary if x['architecture']=='moe_oeo' and x['experts']==n]
        best = max(moe_groups, key=lambda x:x['val_accuracy_mean'])
        d2nn = next(x for x in summary if x['architecture']=='d2nn_expert_global' and x['experts']==n)
        comparisons.append(dict(
            experts=n, selected_top_k=best['top_k'], selection_rule='maximum mean validation accuracy in scanned top-k grid',
            moe_val_accuracy_mean=best['val_accuracy_mean'], moe_val_accuracy_sd=best['val_accuracy_sd'],
            d2nn_val_accuracy_mean=d2nn['val_accuracy_mean'], d2nn_val_accuracy_sd=d2nn['val_accuracy_sd'],
            moe_minus_d2nn_pp=100*(best['val_accuracy_mean']-d2nn['val_accuracy_mean']),
            moe_val_macro_f1_mean=best['val_macro_f1_mean'], d2nn_val_macro_f1_mean=d2nn['val_macro_f1_mean']))
        for seed in [17,27]:
            m=next(r for r in raw if r['architecture']=='moe_oeo' and r['experts']==n and r['top_k']==best['top_k'] and r['seed']==seed)
            d=next(r for r in raw if r['architecture']=='d2nn_expert_global' and r['experts']==n and r['seed']==seed)
            paired.append(dict(experts=n,selected_top_k=best['top_k'],seed=seed,
                               moe_val_accuracy=m['val_accuracy'],d2nn_val_accuracy=d['val_accuracy'],
                               moe_minus_d2nn_pp=100*(m['val_accuracy']-d['val_accuracy'])))

    geometry_rows = []
    for n in [4,16,25,49]:
        mg, mp = geometry_by_arch_n[('moe_oeo',n)]
        dg, dp = geometry_by_arch_n[('d2nn_expert_global',n)]
        target = mg['expert_phase_parameters'] + mg['global_phase_parameters']
        geometry_rows.append(dict(
            experts=n, grid=mg['grid'], expert_side_px=mg['expert_side_px'], gap_px=mg['gap_px'],
            occupied_side_px=mg['occupied_side_px'], active_side_px=mg['active_side_px'],
            canvas_side_px=mg['canvas_side_px'], active_side_mm=mg['active_side_mm'],
            router_side_px=mg['router_side_px'], feature_layers=mg['feature_layers'],
            router_phase_parameters=mg['router_phase_parameters'],
            expert_phase_parameters=mg['expert_phase_parameters'],
            global_phase_parameters=mg['global_phase_parameters'],
            moe_total_parameters=mp, moe_expert_plus_global_parameters=target,
            d2nn_phase_side_px=dg.get('expert_global_actual_parameters') and int(round(math.sqrt(dg['expert_global_actual_parameters']/4))),
            d2nn_parameters=dp, d2nn_relative_error_vs_moe_expert_global=(dp-target)/target))

    raw_fields = list(raw[0])
    summary_fields = list(summary[0])
    route_fields = list(routes[0])
    curve_fields = sorted({key for row in curves for key in row}, key=lambda k:(list(curves[0]).index(k) if k in curves[0] else 999,k))
    confusion_fields = list(confusion[0])
    write_csv(package/'per_seed_runs.csv', raw, raw_fields)
    write_csv(package/'summary_mean_sd.csv', summary, summary_fields)
    write_csv(package/'best_topk_vs_d2nn.csv', comparisons, list(comparisons[0]))
    write_csv(package/'paired_seed_differences.csv', paired, list(paired[0]))
    write_csv(package/'expert_routes.csv', routes, route_fields)
    write_csv(package/'learning_curves.csv', curves, curve_fields)
    write_csv(package/'confusion_matrices.csv', confusion, confusion_fields)
    write_csv(package/'geometry_and_parameters.csv', geometry_rows, list(geometry_rows[0]))

    plotting = dict(
        study='Kather2016 fixed expert size / growing expert count / top-k scan',
        completion=dict(moe_runs=32,d2nn_runs=8,seeds=[17,27],test_read=False),
        standard_deviation='sample standard deviation across seeds (ddof=1)',
        classes=manifest['classes'], split_support=manifest['supports'],
        scan_design=scan_design, source_commit=next(iter(commits)), data_sha256=next(iter(data_hashes)),
        summary=summary, best_topk_vs_d2nn=comparisons, paired_seed_differences=paired,
        geometry=geometry_rows)
    (package/'plotting_summary.json').write_text(json.dumps(plotting,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    lines = [
        '# Kather2016固定专家尺寸与专家数量/Top-k消融', '',
        '本包汇总已完成的32组MoE扫描和8组参数匹配D2NN对照。结果均为验证集，不得标为测试集。', '',
        '## 实验设置', '',
        '- 数据集：Kather2016，8类、5000张150×150 RGB图像，CC BY 4.0。固定划分为train 3496、validation 752、test 752；本实验未读取test。该数据没有可靠患者ID，因此只能声明图像级无重叠划分。',
        '- 随机种子：17、27。误差棒使用种子间样本标准差（ddof=1）；两个种子的标准差只用于展示波动，不是置信区间。',
        '- 单专家尺寸固定224×224，专家间隔30像素。专家数为4、16、25、49，对应2×2、4×4、5×5、7×7排布；完整光学孔径随专家数增加。输入逻辑采样间距17 μm，相位器件映射间距8 μm。',
        '- 每个模型4个主干相位层并逐层加入OEO。MoE交替使用2个专家相位层和2个global相位层；D2NN使用4个连续相位层。没有电子残差、CNN或Qwen分类支路。',
        '- RGB输入分别插值后拼成[R,G;B,mean]单幅振幅图。MoE的224×224编码输入复制到每个专家；D2NN把同一原图直接编码到与其连续相位孔径相同的尺寸，因此完整孔径均受到输入调制。',
        '- 动态路由前向使用hard top-k，反向使用dense振幅straight-through近似。均衡项约束microbatch内soft路由概率均值，权重0.01。top-k只决定入口处的功率分配，后续全场传播允许专家间光场耦合。',
        '- D2NN匹配MoE的专家相位参数与global相位参数之和，不计路由器相位参数；偶数边长取整造成的误差见geometry_and_parameters.csv。',
        '- 训练60轮，AdamW，学习率0.002，effective batch 16、microbatch 2、EMA 0.95。按验证集macro-NLL最小选择checkpoint，因此best checkpoint不一定具有最高验证准确率。', '',
        '## 扫描范围', '',
        '|专家数N|扫描top-k|', '|---:|---|',
        '|4|1, 2, 3, 4|', '|16|1, 4, 8, 16|', '|25|1, 6, 12, 25|', '|49|1, 12, 24, 49|', '',
        '## 主要结果', '',
        '|N|按平均验证准确率选出的k|MoE验证准确率（mean±SD）|D2NN验证准确率（mean±SD）|MoE-D2NN（百分点）|',
        '|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"|{row['experts']}|{row['selected_top_k']}|{row['moe_val_accuracy_mean']*100:.2f}±{row['moe_val_accuracy_sd']*100:.2f}|{row['d2nn_val_accuracy_mean']*100:.2f}±{row['d2nn_val_accuracy_sd']*100:.2f}|{row['moe_minus_d2nn_pp']:.2f}|")
    lines += ['',
        '上述“最佳k”是在同一验证集扫描后选择，仅适合作为探索性结果。正式论文若把它作为主结果，应预先固定k或使用独立验证集选择k，再在test上只评一次。', '',
        '## 文件说明', '',
        '- per_seed_runs.csv：40组训练的逐种子指标，是误差棒和散点的原始数据。',
        '- summary_mean_sd.csv：按架构、N、k汇总的均值和样本标准差。accuracy、balanced accuracy、macro-F1和capture均为0–1，画百分比时乘100；macro-NLL不乘100。',
        '- best_topk_vs_d2nn.csv：每个N按平均验证准确率选出的MoE配置和对应参数匹配D2NN。',
        '- paired_seed_differences.csv：相同N和seed下的MoE-D2NN差值。',
        '- expert_routes.csv：最佳checkpoint的soft路由概率均值与hard选中比例。hard_selection_fraction在专家维求和为k，不能直接画成总和100%的饼图；soft_probability_mean求和为1。',
        '- learning_curves.csv：逐轮训练总目标、验证指标、学习率和路由均衡摘要。训练总目标包含正则项，不等于训练NLL；没有逐轮训练准确率。',
        '- confusion_matrices.csv：长表形式混淆矩阵，行是真实类别，列是预测类别。',
        '- geometry_and_parameters.csv：不同N对应的孔径、参数量和D2NN匹配误差。',
        '- plotting_summary.json：主要汇总的机器可读版本。evidence目录保留原始metadata、history、result、status和phase audit。', '',
        '## 建议作图', '',
        '1. Top-k消融：每个N一个小图，横轴k或k/N，纵轴验证准确率；显示两个seed原始点及mean±SD。',
        '2. 专家数趋势：横轴N，分别画top-1、约25%、约50%、全激活四条曲线。N=4的完整扫描还包含k=3，可只在N=4小图展示。',
        '3. 与D2NN对比：使用best_topk_vs_d2nn.csv画分组点图或柱状图，并在图注中明确“best k selected on validation”。',
        '4. 路由分配：使用expert_routes.csv按seed、N、k画专家选中比例热图；expert_grid_row/col对应专家阵列位置，router_port_row/col对应按中心向外编号的探测端口位置。',
    ]
    (package/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    hashes={}
    for path in sorted(package.iterdir()):
        if path.is_file() and path.name not in {'SHA256.json'}:
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (package/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(raw),'summary_groups':len(summary),'routes':len(routes),
                      'curves':len(curves),'comparisons':comparisons},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
