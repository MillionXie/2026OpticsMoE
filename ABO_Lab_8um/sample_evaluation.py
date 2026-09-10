"""Single fixed-seed query subsample; never select by prediction outcome."""
import argparse
import json
import platform
import random
import subprocess
from pathlib import Path
import numpy as np
from common import sha, read, write


def sample_indices(population, size, seed):
    if not 0 < size <= population: raise ValueError('Invalid subset size')
    return sorted(random.Random(seed).sample(range(population), size))


def metrics(rows):
    ranks = np.array([r['true_rank'] for r in rows], dtype=np.int64)
    if not len(ranks) or (ranks < 1).any(): raise ValueError('Invalid ranks')
    return dict(recall_at_1=float(np.mean(ranks <= 1)), recall_at_5=float(np.mean(ranks <= 5)),
        recall_at_10=float(np.mean(ranks <= 10)), mrr=float(np.mean(1.0/ranks)),
        mean_rank=float(np.mean(ranks)), median_rank=float(np.median(ranks)),
        correct_at_1=int(np.sum(ranks == 1)), n_queries=len(ranks))


def run(source, output, size, seed):
    source, output = Path(source), Path(output)
    report = read(source/'metrics.json')
    rows = report['predictions']
    if len(rows) != report['n_queries']: raise ValueError('Incomplete source predictions')
    indices = sample_indices(len(rows), size, seed)  # Only population size and seed influence membership.
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(type='exploratory_fixed_seed_query_subset', size=size, seed=seed,
        method='Python random.Random(seed).sample(range(N), size), without replacement; sorted by original index',
        one_draw_only=True, selected_using_prediction_outcomes=False, metric_target=None,
        original_queries=len(rows), n_candidates=report['n_candidates'], candidates_unchanged=True,
        source_metrics_sha256=sha(source/'metrics.json'), source_embeddings_sha256=sha(source/'embeddings.npz'),
        checkpoint_sha256=report['checkpoint_sha256'], python=platform.python_version(), numpy=np.__version__,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip())
    write(output/'sampling_protocol.json', protocol)
    chosen = [rows[i] for i in indices]
    selected = set(indices)
    excluded = [r for i,r in enumerate(rows) if i not in selected]
    write(output/'selected_predictions.json', chosen)
    for name, values in [('selected_ids.txt',chosen),('excluded_ids.txt',excluded)]:
        (output/name).write_text('\n'.join(r['sample_id'] for r in values)+'\n',encoding='utf-8')
    with np.load(source/'embeddings.npz', allow_pickle=False) as a:
        if list(a['query_ids']) != [r['sample_id'] for r in rows]: raise ValueError('Embedding order mismatch')
        np.savez_compressed(output/'subset_embeddings.npz', queries=a['queries'][indices],
            labels=a['labels'][indices], query_ids=a['query_ids'][indices],
            titles=a['titles'], title_ids=a['title_ids'], source_query_indices=np.array(indices))
    result = dict(scope='Exploratory subset, NOT the full test set; do not replace full-test metrics',
        subset=metrics(chosen), full_test=metrics(rows), n_candidates=report['n_candidates'], seed=seed)
    write(output/'metrics.json',result)
    (output/'README.md').write_text(
        '# pilot02 固定随机子集（非完整测试集）\n\n'
        f'固定 seed={seed}，从 {len(rows)} 张查询图不放回随机抽取 {size} 张，只抽一次。保留全部 {report["n_candidates"]} 个候选标题。\n\n'
        '不按预测正确/错误筛选，不搜索种子，不以 0.79 或任何指标为目标。原始数据和完整结果未修改。\n\n'
        f'子集 R@1：{result["subset"]["recall_at_1"]:.6f}（{result["subset"]["correct_at_1"]}/{size}）。\n\n'
        f'完整测试 R@1：{result["full_test"]["recall_at_1"]:.6f}（{result["full_test"]["correct_at_1"]}/{len(rows)}）。\n\n'
        '`metrics.json` 同时保留两套指标；`selected_ids.txt`/`excluded_ids.txt` 为选中/未选中名单；'
        '`selected_predictions.json` 保留原查询索引与排名；`subset_embeddings.npz` 包含子集查询及全部候选向量。'
        '这里不复制或删除 CCD 图片。引用时必须注明是随机子集，不应替代完整测试结果。\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('Saved:',output.resolve())


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();run(a.source,a.output,a.size,a.seed)
