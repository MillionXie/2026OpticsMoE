"""CPU-only tables/figures from saved test-sweep evidence; never runs inference."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def arm(path):
    if 'mango_variety_s17_20261010' in path: return '30 epochs'
    if 'mango_variety_s17_e100_lr3' in path: return '100 epochs, LR .003'
    if 'mango_variety_s17_e100_' in path: return '100 epochs, LR .002'
    if 'mango_rho03_continue50_' in path: return 'Continuation (interrupted)'
    if 'requested_gpu1_gpu4' in path: return 'Continuation / migration'
    if 'continue50_round2' in path: return 'Continuation round 2'
    if 'aug_ema' in path: return 'EMA .99' if '/ema/' in path else 'Stronger augmentation'
    if 'capture_smooth' in path: return 'Phase smoothing .05' if '/smooth/' in path else 'Capture loss .05'
    if 'lowlr_ls' in path: return 'Low LR + smoothing' if '/lowlr_ls/' in path else 'Low LR'
    raise ValueError(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    raw = json.loads(a.results.read_text(encoding='utf-8'))
    selection = json.loads(a.selection.read_text(encoding='utf-8'))
    winner = raw['winner']
    assert selection['winner'] == winner
    assert raw['states'] == len(raw['results']) == 36
    assert raw['unique_states'] == len({q['state_sha256'] for q in raw['results']}) == 33
    assert winner['test']['accuracy'] == max(q['test']['accuracy'] for q in raw['results'])
    assert winner == sorted(raw['results'],key=lambda q:(-q['test']['accuracy'],
        q['test']['balanced_nll'],q['checkpoint'],q['state_key']))[0]
    rows = []
    for q in raw['results']:
        tm = q['test']; correct = sum(tm['confusion_matrix'][k][k] for k in range(8))
        assert abs(correct / 418 - tm['accuracy']) < 1e-12
        rows.append(dict(arm=arm(q['checkpoint']), kind=q['kind'], epoch=q['epoch'],
            accuracy=tm['accuracy'], correct=correct, n=418,
            balanced_accuracy=tm['balanced_accuracy'], nll=tm['nll'], balanced_nll=tm['balanced_nll'],
            inference=q['inference'], checkpoint=q['checkpoint'], checkpoint_sha256=q['checkpoint_sha256'],
            state_key=q['state_key'], state_sha256=q['state_sha256'], predictions=q['predictions']))
    with (a.out/'test_sweep.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    compact = {k: v for k, v in raw.items() if k != 'results'}
    compact['results'] = [{k: v for k, v in q.items() if k not in ['config','training_sources']}
                          for q in raw['results']]
    compact['selection_artifact'] = {k:v for k,v in selection.items() if k!='winner'}
    (a.out/'test_sweep.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')
    names = list(dict.fromkeys(q['arm'] for q in rows)); kinds = ['best_ema','last_model','last_ema']
    cells = np.array([[next(q['accuracy'] * 100 for q in rows if q['arm']==name and q['kind']==kind)
                       for kind in kinds] for name in names])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(7.7,6.5))
    im = ax.imshow(cells, cmap='Blues', vmin=70, vmax=90, aspect='auto')
    ax.set_xticks(range(3), ['Best EMA', 'Last raw', 'Last EMA'])
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(3):
            ax.text(j,i,f'{cells[i,j]:.2f}%',ha='center',va='center',
                    color='white' if cells[i,j]>83 else '#18232d')
    ax.set_title('Mango L6, amplitude leakage = 0.3\nTest-selected DEVELOPMENT scores (seed 17)',pad=12)
    fig.colorbar(im,ax=ax,label='Accuracy (%)',shrink=.75)
    fig.text(.03,.015,'36 saved states / 33 unique weights; 418 test images. Extra optimization budget.\n'
             'Not independent generalization or an equal-budget residual ablation.',fontsize=8)
    fig.tight_layout(rect=(0,.075,1,1))
    for ext in ['png','svg','pdf']: fig.savefig(a.out/f'test_sweep.{ext}',dpi=180)
    plt.close(fig)
    cfg = winner['config']; tm = winner['test']
    correct = sum(tm['confusion_matrix'][k][k] for k in range(8))
    lines = ['# 六层30%残差：测试集选模开发扫描', '',
        '2026-10-10用户明确授权直接在test上选PT。本轮没有训练或集成，也未修改数据、光路、残差比例、类别或种子。',
        '**结果为测试集选模的开发成绩，不能称独立泛化或同预算公平消融；原验证选模报告保留。**', '',
        f'最高准确率 **{tm["accuracy"]*100:.2f}%（{correct}/418）**，宏平均召回{tm["balanced_accuracy"]*100:.2f}%，'
        f'balanced NLL {tm["balanced_nll"]:.6f}。获选{arm(winner["checkpoint"])}的{winner["kind"]}，第{winner["epoch"]}轮。',
        '87.80%有三组权重同分，按预先写入manifest的“同分最低test balanced NLL”选择相位平滑组。',
        '比此前验证选定EMA的87.08%高0.72pp；比固定无残差87.56%高0.24pp（仅一个样本净差，不是显著优势）。',
        '89.56%以上需至少375/418=89.71%，当前还差8个正确样本。此值只是已保存权重中的最高值，不能当理论上限。', '',
        f'选定训练设置：lr={cfg["lr"]}余弦，EMA={cfg["ema_decay"]}，相位平滑={cfg["phase_smooth_weight"]}，'
        f'收光损失={cfg["capture_weight"]}，label smoothing={cfg["label_smoothing"]}；九专家、3专家层+3global，router无残差。', '',
        '| 保存训练臂 | best EMA | last raw | last EMA |', '|---|---:|---:|---:|']
    for i, name in enumerate(names):
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}%' for v in cells[i])+' |')
    lines += ['', '扫描12个训练臂、36个状态、33组不同权重。新增'+str(raw['new_inferences'])+
              '组测试推理，其余复用历史或本轮同状态记录；原中断run明确保留中断身份，不冒充完整训练。',
        'best表示当时验证选中的EMA；last raw表示末轮训练权重；last EMA是同一个last PT中保存的EMA。没有保存逐轮PT，因此不能扫描每一轮。',
        '逐样本ID、标签支持数、argmax、混淆矩阵、正确数及指标一致性均已检查。源码身份校验原已发布Git blob，未改旧PT的sources。', '',
        '## 权重与证据', '',
        f'- 原checkpoint：`{winner["checkpoint"]}`',
        f'- 原checkpoint SHA256：`{winner["checkpoint_sha256"]}`',
        f'- state key：`{winner["state_key"]}`，state SHA256：`{winner["state_sha256"]}`',
        f'- 开发选定导出：`{selection["selected_artifact"]}`',
        f'- 导出SHA256：`{selection["selected_artifact_sha256"]}`',
        f'- 源训练版本：`{winner["training_source_revision"]}`；完整config/sources保留服务器manifest与results。',
        f'- manifest SHA256：`{raw["manifest_sha256"]}`',
        '- [36格CSV](test_sweep.csv)、[JSON证据](test_sweep.json)、[比较图](test_sweep.png)、[SVG](test_sweep.svg)、[PDF](test_sweep.pdf)。',
        '- 测试418条，单种子，图像级划分；无误差条。选模反复接触此集合会使最高值乐观，后续独立泛化须用新保留数据。',
        '- 本次仅使用指定物理GPU4，进程完成后退出并释放。无残差和2/4层权重不变。', '']
    (a.out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(winner_accuracy=tm['accuracy'],correct=correct,
                          states=raw['states'],new_inferences=raw['new_inferences'])))


if __name__=='__main__': main()
