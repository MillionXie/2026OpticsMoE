"""Audit the sixteen completed T25 simulations and render paired results and optical diagnostics."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from LightGenV2.tasks.t24_continuous_spatial_routing.compare import DATASETS, SPLITS, inspect_run, read_json, require
from LightGenV2.tasks.t24_continuous_spatial_routing.plot_epoch_curves import split_metrics
from LightGenV2.tasks.t24_continuous_spatial_routing.stability import assess_run


TASK = Path(__file__).resolve().parent
PROFILE = TASK / 'configs' / 'matched_fiber_20261128.json'
ARMS = {'diffraction': '4 diffraction', 'mixed': '2 diffraction + 2 fiber', 'uniform_half': '4 half-fiber', 'fiber': '4 fiber'}
CHINESE_ARMS = {'diffraction': '四衍射', 'mixed': '两衍射＋两光纤', 'uniform_half': '四槽各半光纤', 'fiber': '四光纤'}
CONTRASTS = (('mixed', 'uniform_half'), ('mixed', 'diffraction'), ('mixed', 'fiber'), ('fiber', 'diffraction'))
CORNERS = np.array([[0, 0], [0, 254], [254, 0], [254, 254]])
TOL = 2e-5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_matches(path, expected):
    content = path.read_bytes()
    return any(hashlib.sha256(b).hexdigest() == expected for b in (content, content.replace(b'\r\n', b'\n')))


def verified_gpu(meta, config, profile, seed):
    actual = meta.get('gpu_uuid')
    require(isinstance(actual, str) and actual.startswith('GPU-')
            and actual == config.get('assigned_gpu_uuid') == profile['gpu_uuids'].get(str(seed)),
            f'Seed {seed}: observed GPU UUID differs from configuration or protocol assignment')
    return actual


def optical_usage(path, fibers, n):
    with np.load(path, allow_pickle=False) as z:
        positions, rank, raw, dense = (z[k] for k in ('positions', 'slot_indices', 'router_port_energy', 'dense_power'))
        capture, power, gate, before, after = (z[k] for k in
            ('router_capture', 'power', 'selected_gate_power', 'pre_fiber_power', 'transmitted_power'))
    require(rank.shape == (n, 2) and np.issubdtype(rank.dtype, np.integer)
            and np.all((rank >= 0) & (rank < 4)) and np.all(rank[:, 0] != rank[:, 1])
            and np.array_equal(positions, CORNERS[rank]), f'{path}: actual two distinct corners disagree with slots')
    idx = np.arange(n)[:, None]
    for x in (raw, dense, gate, before, after):
        require(x.shape == (n,4) and np.isfinite(x).all() and (x >= 0).all(), f'{path}: invalid four-slot power')
    chosen = np.zeros((n,4), dtype=bool)
    chosen[idx, rank] = True
    ranked = raw[idx, rank]
    require(np.allclose(dense, (raw + 1e-12)/(raw.sum(1, keepdims=True)+4e-12), atol=3e-7, rtol=2e-5)
            and np.all(dense[idx, rank][:,1] >= np.where(chosen, -1, dense).max(1))
            and np.all(dense[idx, rank][:,0] >= dense[idx, rank][:,1]), f'{path}: gate disagrees with CCD Top-2')
    require(power.shape == (n,2) and capture.shape == (n,) and (power > 0).all()
            and np.allclose(power.sum(1), 1., rtol=0., atol=TOL)
            and np.allclose(capture, ranked.sum(1), atol=TOL, rtol=0)
            and np.allclose(power, (ranked+1e-12)/(capture[:,None]+2e-12), atol=TOL, rtol=0)
            and np.allclose(gate[idx, rank], power, atol=TOL, rtol=0) and (gate[~chosen] == 0).all(),
            f'{path}: recorded selected powers disagree')
    incident_total = before.sum(1, keepdims=True)
    require(np.allclose(before, gate*incident_total, atol=3e-6, rtol=2e-4)
            and np.all(np.isclose(incident_total, 1., atol=2e-4) | (incident_total == 0.))
            and np.all(after <= before+TOL) and np.all(after.sum(1) <= before.sum(1)+TOL)
            and (after[~chosen] <= TOL).all(), f'{path}: unphysical incident power, dark slot, or transmission gain')
    ordinary = [i for i in range(4) if i not in fibers]
    require(np.allclose(before[:,ordinary], after[:,ordinary], atol=TOL, rtol=0),
            f'{path}: unfiltered diffraction slot lost energy')
    incident, transmitted = before.astype(float).mean(0), after.astype(float).mean(0)
    counts = chosen.sum(0)
    pairs, amounts = np.unique(np.sort(rank, axis=1), axis=0, return_counts=True)
    return {'n': n, 'selection_count': counts.tolist(), 'inclusion_fraction': (counts/n).tolist(),
            'fraction_of_two_selections': (counts/(2*n)).tolist(),
            'pair_counts': {f'E{a+1}+E{b+1}': int(v) for (a,b), v in zip(pairs, amounts)},
            'mean_dense_router_share': dense.astype(float).mean(0).tolist(),
            'mean_gate_power_after_top2': gate.astype(float).mean(0).tolist(),
            'mean_incident_fraction': incident.tolist(), 'mean_transmitted_fraction': transmitted.tolist(),
            'conditional_power_transmission': [float(b/a) if a>0 else None for a,b in zip(incident,transmitted)],
            'zero_input_images': int(np.count_nonzero(incident_total==0)),
            'total_input_fraction': float(incident.sum()), 'total_transmitted_fraction': float(transmitted.sum()),
            'total_transmittance': float(transmitted.sum()/incident.sum()) if incident.sum()>0 else None,
            'max_observed_slot_gain': float(np.max(after-before)), 'mean_router_capture': float(capture.mean())}


def checked_run(directory, dataset, seed, arm, profile):
    meta, history, arrays, _ = inspect_run(directory, seed, arm, dataset)
    manifest = read_json(directory/'manifest.json')
    for name, digest in manifest.items():
        require(Path(name).name == name and sha(directory/name) == digest, f'{directory.name}/{name}: bad manifest')
    result, config = read_json(directory/'result.json'), read_json(directory/'config.json')
    gpu_uuid = verified_gpu(meta, config, profile, seed)
    expected_fibers = profile['fiber_slots_zero_based'][arm]
    if arm == 'mixed':
        expected_fibers = expected_fibers[str(seed)]
    cores = [392 if arm=='uniform_half' else 784 if i in expected_fibers else 0 for i in range(4)]
    status = read_json(directory/'status.json')
    require('config.json' in manifest and meta['arm'] == result['arm'] == arm and config == meta['config']
            and meta['protocol_sha256'] == config['protocol_sha256']
            and source_matches(PROFILE, meta['protocol_sha256'])
            and config['fiber_slots'] == expected_fibers and meta['fiber_core_counts'] == config['fiber_core_counts'] == cores
            and sum(cores) == profile['fiber_channels'][arm] and config['fiber_model'] == profile['fiber_model']
            and meta['config_file_sha256'] == manifest['config.json'] and not meta['smoke']
            and result['diagnostic_TEST_calls'] == result['target_epoch'] == profile['epochs']
            and status['epoch'] == status['target_epoch'] == profile['epochs']
            and status['selected_epoch'] == result['selected_epoch']
            and meta['fiber_slots'] == expected_fibers and config['epochs'] == profile['epochs'] == len(history)
            and meta['parameter_count'] == 513416, f'{directory.name}: profile, layout, or model size differs')
    base_file = TASK.parent / profile['base_configs'][dataset]
    require(source_matches(base_file, config['data_config_sha256']), 'Archived base profile checksum differs')
    base = read_json(base_file)
    for key in ('raw_cache_sha256','manifest_sha256','index_sha256','source_archive_sha256','expected_counts'):
        require(config[key] == base[key], f'{directory.name}: input dataset identity differs at {key}')
    require(all(config[k] == profile['training'][k] for k in ('batch','phase_lr','head_lr','label_smoothing')),
            f'{directory.name}: training hyperparameters differ')
    n = config['expected_counts']
    require(meta['data']['counts'] == n and meta['data']['manifest_sha256'] == config['manifest_sha256']
            and meta['data']['index_sha256'] == config['index_sha256']
            and meta['data']['archive_sha256'] == config['source_archive_sha256'], 'Evaluated data differs from config')
    rng = np.random.default_rng(seed)
    rows = []
    for e in history:
        order = rng.permutation(n['train']).astype(np.int64)
        require(hashlib.sha256(order.tobytes()).hexdigest() == e['minibatch_order_sha256']
                and np.isfinite(e['train_loss']), f'{directory.name} E{e["epoch"]}: unexpected order or invalid loss')
        row = {'epoch': e['epoch'], 'online_train_loss': e['train_loss']}
        for s in SPLITS:
            m = e['validation' if s == 'val' else s]
            require(np.isfinite(m['cross_entropy']) and m['cross_entropy'] >= 0, 'Invalid observed cross entropy')
            row[s] = {**split_metrics(m, n[s]), 'cross_entropy': m['cross_entropy']}
        rows.append(row)
    best = result['selected_epoch']
    for s in SPLITS:
        require(rows[best-1][s]['correct'] == result[s]['correct'], f'{directory.name}/{s}: locked outcome mismatch')
    usage = {s: optical_usage(directory/f'predictions_{s}.npz', expected_fibers, n[s]) for s in SPLITS}
    metadata = {'run': directory.name, 'source_commit': meta['source_commit'], 'arm': arm, 'seed': seed, 'gpu_uuid': gpu_uuid,
                'fiber_slots_zero_based': expected_fibers, 'fiber_core_counts': cores, 'parameter_count': meta['parameter_count'],
                'initial_state_sha256': meta['initial_state_sha256'], 'data': meta['data'],
                'optimizer': meta['optimizer'], 'source_sha256': meta['source_sha256'],
                'manifest_sha256': sha(directory/'manifest.json'), 'artifact_sha256': manifest,
                'selected_epoch': best, 'test_selection_calls': result['TEST_selection_calls'],
                'epochs': rows, 'locked': {s: rows[best-1][s] for s in SPLITS},
                'final': {s: rows[-1][s] for s in SPLITS}, 'locked_optics': usage,
                'optimizer_steps': sum(r['train_steps'] for r in history),
                'minibatch_orders_sha256': hashlib.sha256(b''.join(bytes.fromhex(e['minibatch_order_sha256'])
                                              for e in history)).hexdigest(), 'val_plateau_at_endpoint': assess_run(history)}
    return metadata, arrays


def section_summary(runs, predictions, seeds):
    aligned = {(r['seed'], r['arm']): r for r in runs}
    anchor = next(iter(predictions.values()))
    for name, arrays in predictions.items():
        for s in SPLITS:
            require(np.array_equal(arrays[s][0], anchor[s][0]) and np.array_equal(arrays[s][1], anchor[s][1]),
                    f'{name}/{s}: unmatched original IDs or labels')
    require(all(r['data'] == runs[0]['data'] and r['optimizer'] == runs[0]['optimizer']
                and r['optimizer_steps'] == runs[0]['optimizer_steps'] and r['source_commit'] == runs[0]['source_commit']
                and r['source_sha256'] == runs[0]['source_sha256'] for r in runs), 'Unmatched data, optimizer, code, or budget')
    differences = []
    gpu_by_seed = {}
    for seed in seeds:
        rows = {r['arm']: r for r in runs if r['seed'] == seed}
        require(set(rows) == set(ARMS) and all(len({r[k] for r in rows.values()}) == 1 for k in
                ('initial_state_sha256','minibatch_orders_sha256')), f'Seed {seed} mismatched initialization/order')
        gpus = {r['gpu_uuid'] for r in rows.values()}
        require(len(gpus) == 1, f'Seed {seed}: paired arms used different GPU UUIDs')
        gpu_by_seed[seed] = gpus.pop()
    for candidate, reference in CONTRASTS:
        per_seed = {}
        for seed in seeds:
            a,b = (aligned[seed,k] for k in (candidate,reference))
            pair = {}
            for s in SPLITS:
                good_a, good_b = predictions[a['run']][s][2], predictions[b['run']][s][2]
                wins, losses = int((good_a & ~good_b).sum()), int((~good_a & good_b).sum())
                total = len(good_a)
                end_delta = a['final'][s]['correct'] - b['final'][s]['correct']
                pair[s] = {'n': total, 'locked_candidate_only_correct': wins, 'locked_reference_only_correct': losses,
                           'locked_net_correct': wins-losses, 'locked_delta_pp': 100*(wins-losses)/total,
                           'final_net_correct': end_delta, 'final_delta_pp': 100*end_delta/total}
            per_seed[seed] = pair
        averages = {k: {s: float(np.mean([p[s][k] for p in per_seed.values()])) for s in SPLITS}
                    for k in ('locked_delta_pp', 'final_delta_pp')}
        differences.append({'candidate': candidate, 'reference': reference, 'seeds': per_seed, **averages})
    mean = {}
    for arm in ARMS:
        group = [r for r in runs if r['arm']==arm]
        mean[arm] = {stage: {s: float(np.mean([r[stage][s]['accuracy_percent'] for r in group])) for s in SPLITS}
                     for stage in ('locked','final')}
    return {'equal_weight_seed_means_percent': mean, 'paired_differences': differences, 'matched_gpu_uuid_by_seed': gpu_by_seed}


def save(fig, root):
    svg = root.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date': None})
    lines = svg.read_text(encoding='utf-8').splitlines()
    svg.write_bytes(('\n'.join(l.rstrip() for l in lines)+'\n').encode('utf-8'))
    fig.savefig(root.with_suffix('.png'), dpi=135)
    plt.close(fig)


def figures(title, runs, seeds, prefix):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','svg.hashsalt':'t25-fig-v1'})
    fig, axes = plt.subplots(len(seeds), len(ARMS), figsize=(14.1,6.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07,right=.982,top=.84,bottom=.11,hspace=.28,wspace=.12)
    lines = {'train':('#087368','-'), 'val':('#405783','--'), 'test':('#AB5218','-')}
    for r in runs:
        i,j = seeds.index(r['seed']), list(ARMS).index(r['arm'])
        ax = axes[i,j]
        epochs = [e['epoch'] for e in r['epochs']]
        for s,(color,ls) in lines.items():
            ax.plot(epochs, [e[s]['accuracy_percent'] for e in r['epochs']], c=color, ls=ls, lw=1.12)
        ax.scatter(r['selected_epoch'], r['locked']['val']['accuracy_percent'], marker='*', s=105, color=lines['val'][0],
                   edgecolor='white', lw=.5, zorder=5)
        ax.scatter(epochs[-1],r['final']['test']['accuracy_percent'], marker='s', s=22, color=lines['test'][0], zorder=4)
        ax.set_title(f'{ARMS[r["arm"]]} · seed {r["seed"]}', loc='left', fontweight='semibold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_xlim(1,epochs[-1]+2)
        if j==0: ax.set_ylabel('Accuracy (%)')
        if i==len(seeds)-1: ax.set_xlabel('Epoch')
    fig.suptitle(f'{title}  |  End-of-epoch accuracy', fontsize=16, fontweight='semibold', y=.975)
    legend = [Line2D([],[],c=c,ls=ls,label=s.upper()) for s,(c,ls) in lines.items()]
    legend += [Line2D([],[],ls='',marker=m,c=c,label=t) for m,c,t in
               [('*',lines['val'][0],'Best VAL'), ('s',lines['test'][0],'Final TEST')]]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,.93), ncol=5, frameon=False)
    save(fig, prefix.parent/(prefix.name+'_trajectories'))

    fig, axes = plt.subplots(1,3,figsize=(11.7,5.0))
    fig.subplots_adjust(left=.168,right=.965,top=.76,bottom=.20,wspace=.13)
    measures = [('inclusion_fraction','Images selecting slot (%)'),
                ('mean_incident_fraction','Incident slot power (%)'),
                ('mean_transmitted_fraction','Transmitted slot power (%)')]
    labels = [f'{ARMS[r["arm"]]} · s{r["seed"]}' for r in runs]
    for j,(key,label) in enumerate(measures):
        ax = axes[j]
        a = 100*np.array([r['locked_optics']['test'][key] for r in runs])
        ax.imshow(a, cmap='Blues', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(4),[f'E{i+1}' for i in range(4)])
        ax.set_yticks(range(len(labels)),labels if j==0 else ['']*len(labels))
        ax.tick_params(length=0)
        ax.set_title(label, fontsize=9.5)
        for row in range(len(runs)):
            for col in range(4):
                ax.text(col,row,f'{a[row,col]:.1f}',ha='center',va='center',color='white' if a[row,col]>55 else '#182333')
    fig.suptitle(f'{title}  |  TEST routing and optical power at each VAL-selected checkpoint',
                 y=.96,fontweight='semibold',fontsize=13)
    fig.text(.17,.08,'Two slots per image. Power columns are relative to total incident power before the expert filter and OEO.',
             fontsize=9)
    save(fig,prefix.parent/(prefix.name+'_optics'))


def table(headings, rows):
    esc = lambda value: html.escape(str(value))
    return ('<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headings)
            +'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in r)+'</tr>' for r in rows)
            +'</tbody></table></div>')


def html_report(report):
    css = ('body{font:15px/1.6 system-ui,sans-serif;color:#20302d;margin:26px auto;max-width:1140px;padding:0 16px}'
           'h1,h2,h3{line-height:1.25}h2{border-top:1px solid #c5cdca;padding-top:14px}table{border-collapse:collapse}'
           'th,td{padding:4px 10px;text-align:right;border-bottom:1px solid #dce2df;white-space:nowrap}td:first-child,th:first-child'
           '{text-align:left}th{background:#eef3f0}.scroll{overflow:auto}img{width:100%;height:auto}nav a{margin-right:18px}'
           'summary{cursor:pointer;font-weight:600}p{max-width:95ch}a{color:#176751}')
    chunks = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
              '<title>T25 光纤专家仿真对照</title><style>'+css+'</style><h1>固定四专家 Top-2 的光纤阵列仿真对照</h1>',
              '<p>每个数据集使用种子 37 和 53。每种配置均从相同的初始可训练参数开始，记录 180 轮完整训练、验证与测试准确率。'
              '最佳权重按 VAL 正确数、VAL 交叉熵和较早轮次依次选择。末轮表格使用同一轮的实际权重。</p>',
              '<p>主要对照将相同的 1,568 个单模通道集中在两个槽（每槽 784 个），或均匀分布到四个槽'
              '（每槽 392 个，棋盘格内其余位置直接透射）。两组之外另有四衍射和四光纤对照。</p>',
              '<nav><a href="#kather">Kather2016</a><a href="#blood">BloodMNIST128</a><a href="results.json">可复算数据与 SHA</a></nav>']
    number = lambda x: f'{x:.2f}%'
    for ds,data in report['datasets'].items():
        summary, runs = data['summary'], data['runs']
        chunks.append(f'<section id="{ds}"><h2>{DATASETS[ds][0]}</h2><h3>两个种子的平均准确率</h3>')
        cols = ['配置','VAL 选模 TRAIN','VAL 选模 VAL','VAL 选模 TEST','E180 TRAIN','E180 VAL','E180 TEST']
        means = summary['equal_weight_seed_means_percent']
        chunks.append(table(cols, [[CHINESE_ARMS[a],*[number(means[a][stage][s]) for stage in ('locked','final')
                                                      for s in SPLITS]] for a in ARMS]))
        chunks.append('<h3>同种子 TEST 准确率差（百分点）</h3>')
        contrasts = []
        for pair in summary['paired_differences']:
            p = pair['seeds']
            contrasts.append([CHINESE_ARMS[pair['candidate']]+' − '+CHINESE_ARMS[pair['reference']],
                              *[f'{p[s]["test"]["locked_delta_pp"]:+.2f}' for s in (37,53)],
                              f'{pair["locked_delta_pp"]["test"]:+.2f}',
                              *[f'{p[s]["test"]["final_delta_pp"]:+.2f}' for s in (37,53)],
                              f'{pair["final_delta_pp"]["test"]:+.2f}'])
        chunks.append(table(['配置差','选模 s37','选模 s53','选模平均','E180 s37','E180 s53','E180 平均'],contrasts))
        chunks.append('<details><summary>逐个模型的准确率、选中轮次及末轮稳定性</summary>')
        rows = [[CHINESE_ARMS[r['arm']],r['seed'],r['selected_epoch'],
                 *[number(r[st][s]['accuracy_percent']) for st in ('locked','final') for s in SPLITS],
                 '满足' if r['val_plateau_at_endpoint']['plateau'] else '未满足'] for r in runs]
        chunks += [table(['配置','种子','最佳轮次',*cols[1:],'E180 VAL 平台规则'],rows),'</details>',
                   f'<img src="{ds}_trajectories.svg" alt="{DATASETS[ds][0]} 八组实测逐轮准确率">',
                   '<h3>真实入选频率与模拟通光率</h3><p>每幅图选两个槽，四项入选频率合计为 200%。'
                   '入射与透过功率均除以专家平面总入射功率，以 OEO 前的复光场计算。混合组在种子 37 用 E1/E4，'
                   '在种子 53 用 E2/E3。四槽各半组的出射含导模分量和直接透射分量。图中同一行均来自该行的 VAL 选中权重与同一批 TEST 图像。</p>',
                   f'<img src="{ds}_optics.svg" alt="{DATASETS[ds][0]} 专家入选频率和入射、出射功率">',
                   table(['配置','种子','TRAIN 总通光率','VAL 总通光率','TEST 总通光率'],
                         [[CHINESE_ARMS[r['arm']],r['seed'],*[number(100*r['locked_optics'][s]['total_transmittance'])
                                                             for s in SPLITS]] for r in runs]),'</section>']
    chunks.append('<p>平台判断使用 T24 的三个 VAL 阈值，仅用于描述固定预算末轮。两种子的混合组同时包含初始化和对角布局差异。'
                  '各配置可训练参数、样本和更新次数匹配，光学秩、透光率与算子成本随配置变化。原 TEST 曾用于 T17 开发。'
                  'Blood 原划分有 5 个相同像素哈希跨 TRAIN/TEST、1 个跨 VAL/TEST。</p></html>')
    return ''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True,help='Parent of all sixteen completed T25 runs')
    parser.add_argument('--out',type=Path,required=True,help='Report output directory')
    args = parser.parse_args()
    profile = read_json(PROFILE)
    require(profile['seeds'] == [37,53] and profile['epochs'] == 180 and set(profile['arms']) == set(ARMS)
            and set(profile['datasets']) == set(DATASETS), 'Unexpected cohort profile')
    datasets = {}
    for ds in profile['datasets']:
        runs, predictions = [], {}
        for seed in profile['seeds']:
            for arm in profile['arms']:
                name = f'{ds}_s{seed}_{arm}'
                run, predictions[name] = checked_run(args.runs/name, ds, seed, arm, profile)
                runs.append(run)
        datasets[ds] = {'runs': runs, 'summary': section_summary(runs,predictions,profile['seeds'])}
    args.out.mkdir(parents=True,exist_ok=True)
    for ds, data in datasets.items():
        figures(DATASETS[ds][0],data['runs'],profile['seeds'],args.out/ds)
        data['figure_sha256'] = {p.name: sha(p) for kind in ('trajectories','optics') for ext in ('svg','png')
                                  for p in [args.out/f'{ds}_{kind}.{ext}']}
    report = {'profile': profile, 'profile_sha256':sha(PROFILE), 'report_source_sha256':sha(Path(__file__)),
              'datasets':datasets, 'accuracy_units':'percent, with comparisons in percentage points',
              'optical_units':'Fractions, equal-weight means over split images; pre/post powers refer to unit incident '
                  'power before fiber projection and OEO. Conditional transmittance is ratio of summed post/pre powers.',
              'epoch_budget':180, 'selection':'Best VAL, then lower VAL CE, then earliest exact tie; TEST is diagnostic.',
              'stability':'T24 VAL-only diagnostic at E180 on the actual fixed-budget T25 trajectory.'}
    (args.out/'results.json').write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
    (args.out/'results.html').write_bytes((html_report(report)+'\n').encode())


if __name__ == '__main__':
    main()
