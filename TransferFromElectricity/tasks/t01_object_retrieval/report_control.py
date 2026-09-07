"""Summarize controlled development and three-dataset fixed-bank experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


LABELS = {'fixed': 'Fixed experts', 'direct': 'Direct phase', 'qwen_frozen': 'Frozen Qwen',
          'qwen_lora': 'Qwen LoRA', 'clip_frozen': 'Frozen CLIP', 'clip_lora': 'CLIP LoRA'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(run):
    if read(run / 'status.json')['status'] != 'complete':
        raise ValueError(f'Incomplete run: {run}')
    cfg, result = read(run / 'protocol.json'), read(run / 'final_report.json')
    env, history = read(run / 'environment.json'), read(run / 'history.json')
    if 'RTX' not in env['device'] or 'A100' in env['device']:
        raise ValueError('This comparison requires RTX runs')
    if result['export_max_error'] != 0 or any(h['frozen_parameter_max_change'] != 0 for h in history):
        raise ValueError('Export or freezing audit failed')
    return cfg, result, env, history


def development(runs):
    rows, contracts, expected = [], set(), set()
    for run in runs:
        cfg, result, env, history = load(run)
        if result['evaluation_split'] != 'validation' or result['selected_live_test'] is not None:
            raise ValueError('Development must not inspect test metrics')
        if cfg['method'] != 'qwen_lora':
            raise ValueError('Development grid is Qwen only')
        if not env.get('deterministic_algorithms', False):
            raise ValueError('Deterministic development runs are required')
        last = history[-3:]
        candidate = not cfg.get('freeze_electronic', False) and 'decoder_freeze_after' not in cfg
        row = {'run_id': run.name, 'candidate': candidate,
               'electronic_lr': cfg['learning_rates']['electronic'], 'generator_lr': cfg['learning_rates']['generator_context'],
               'freeze_electronic': cfg.get('freeze_electronic', False), 'decoder_freeze_after': cfg.get('decoder_freeze_after'),
               'last3_validation_top1': statistics.mean(h['live_validation']['top1_retrieval_accuracy'] for h in last),
               'last3_validation_mrr': statistics.mean(h['live_validation']['mrr'] for h in last),
               'selected_validation_top1': result['selected_metrics']['top1_retrieval_accuracy'],
               'selected_validation_top3': result['selected_metrics']['top3_retrieval_accuracy'],
               'phase_rms_rad': result['selected_expert_phase']['rms_change_rad'],
               'lora_phase_effect_rad': result['ablations']['lora_phase_effect']['rms_change_rad'],
               'ablations': {k: v['top1_retrieval_accuracy'] for k,v in result['ablations'].items() if 'top1_retrieval_accuracy' in v}}
        rows.append(row)
        if candidate:
            common = json.loads(json.dumps(cfg))
            for key in ('electronic','readout','generator_context'): common['learning_rates'].pop(key)
            contracts.add((result['git_sha'], result['split_sha256'], env['device'], json.dumps(common,sort_keys=True)))
            expected.add((row['electronic_lr'], row['generator_lr']))
    if len(contracts) != 1 or expected != {(e,g) for e in (1e-4,1e-5) for g in (1e-4,1e-3)}:
        raise ValueError('The complete paired 2x2 learning-rate grid is required')
    chosen = max((r for r in rows if r['candidate']), key=lambda r: (r['last3_validation_top1'],r['last3_validation_mrr']))
    return {'rows': rows, 'chosen': chosen, 'selection': 'last-three validation Top-1 mean then MRR mean; no test evaluation'}


def summarize(runs, dev_runs, output):
    output.mkdir(parents=True, exist_ok=True)
    evidence, rows, contracts = [], [], {}
    for run in runs:
        cfg, result, env, history = load(run)
        if result['evaluation_split'] != 'test': raise ValueError('Formal report requires test evaluation')
        if not env.get('deterministic_algorithms', False): raise ValueError('Deterministic formal runs are required')
        dataset = cfg.get('dataset', {}).get('name', 'caltech')
        key = (dataset, result['method'])
        if key in contracts: raise ValueError(f'Duplicate {key}')
        comparable = {k:v for k,v in cfg.items() if k != 'method'}
        contracts[key] = (result['git_sha'], result['split_sha256'], env['device'], json.dumps(comparable,sort_keys=True))
        m = result['selected_metrics']
        ablations = {k:{name:v[name] for name in ('top1_retrieval_accuracy','top3_retrieval_accuracy','mrr')}
                     for k,v in result['ablations'].items() if 'top1_retrieval_accuracy' in v}
        minimum = min(v for h in history for a in h['fusion'].values() for v in a)
        maximum = max(v for h in history for a in h['fusion'].values() for v in a)
        if not .59999 < minimum <= maximum < .60001: raise ValueError('Expected frozen alpha=0.6')
        optical_drop = 100*(m['top1_retrieval_accuracy']-ablations['remove_optical']['top1_retrieval_accuracy'])
        row = {'dataset': dataset, 'method': result['method'], 'run_id': run.name,
               'top1_percent': 100*m['top1_retrieval_accuracy'], 'top3_percent': 100*m['top3_retrieval_accuracy'], 'mrr': m['mrr'],
               'selected_epoch': result['selected_epoch'], 'phase_rms_rad': result['selected_expert_phase']['rms_change_rad'],
               'optical_removal_top1_drop_pp': optical_drop,
               'optical_dependency_5pp_pass': optical_drop >= 5.,
               'electronic_removal_top1_drop_pp': 100*(m['top1_retrieval_accuracy']-ablations['remove_electronic']['top1_retrieval_accuracy']),
               'vision_optical_removal_top1_drop_pp': 100*(m['top1_retrieval_accuracy']-ablations['remove_vision_optical']['top1_retrieval_accuracy']),
               'language_optical_removal_top1_drop_pp': 100*(m['top1_retrieval_accuracy']-ablations['remove_language_optical']['top1_retrieval_accuracy']),
               'counts': result['counts'], 'ablations': ablations,
               'lora_phase_effect_rad': result['ablations'].get('lora_phase_effect',{}).get('rms_change_rad'),
               'peak_memory_gib': result['peak_memory_gib'], 'elapsed_seconds': result['elapsed_seconds'],
               'device': env['device'], 'gpu_uuid': env['cuda_visible_devices'],
               'git_sha': result['git_sha'], 'updates': result['optimizer_updates'],
               'alpha_min': minimum, 'export_max_error': result['export_max_error'],
               'trainable_parameter_counts': {g['group_name']:g['parameter_count'] for g in read(run/'architecture.json')['optimizer_groups']},
               'last_routing_counts': history[-1]['expert_selection_counts']}
        rows.append(row)
    for dataset in {k[0] for k in contracts}:
        if len({v for k,v in contracts.items() if k[0]==dataset}) != 1:
            raise ValueError(f'Mismatched method contracts: {dataset}')
        required = {'fixed','direct','qwen_lora','clip_lora'} | ({'qwen_frozen','clip_frozen'} if dataset=='cifar100' else set())
        if {k[1] for k in contracts if k[0]==dataset} != required:
            raise ValueError(f'Incomplete method set: {dataset}')
    if runs and {k[0] for k in contracts} != {'caltech','cifar100','imagenette'}:
        raise ValueError('All three datasets are required')
    for run in runs + dev_runs:
        for name in ('protocol.json','final_report.json','environment.json','history.json','steps.json','initialization.json','architecture.json','expert_bank.pt','transfer_manifest.json'):
            path=run/name
            if path.exists(): evidence.append({'run_id':run.name,'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'formal': rows, 'development': development(dev_runs),
              'limitations': ['One optimization seed for this protocol; no statistical significance claim',
                              'Caltech historical warmstart and prior test exposure',
                              'Class-prototype retrieval with held-out official evaluation images; not classifier-head benchmark accuracy',
                              'Branch removals are post-training interventions, not independently trained pure optical/electronic models',
                              'Ideal optical simulation; no hardware robustness guarantee',
                              'Different pretrained encoder sizes and trainable parameter counts; timing on a shared server']}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    if rows:
        result['phase_diagnostics'] = phase_diagnostics(runs,output)
        (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        flat = [{k:v for k,v in row.items() if k not in {'counts','ablations','last_routing_counts','trainable_parameter_counts'}} for row in rows]
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        plot(rows,output)
    training_diagnostics(runs, dev_runs, output)
    if rows:
        write_results_markdown(result,output)
    return result


def write_results_markdown(result,output):
    names={'caltech':'Caltech 十类','cifar100':'CIFAR-100 固定十类','imagenette':'Imagenette 十类'}
    lines=['# 三数据集固定专家库：完整结果','',
           '由完整 run 自动生成。Top-1 / Top-3 为类别原型检索，checkpoint 只按验证集选择；本轮每组一个优化 seed。','',
           '| 数据集 | 方法 | Top-1 | Top-3 | MRR | 所选轮次 | 相位 RMS / rad |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in result['formal']:
        lines.append(f"| {names[r['dataset']]} | {LABELS[r['method']]} | {r['top1_percent']:.2f}% | {r['top3_percent']:.2f}% | {r['mrr']:.4f} | {r['selected_epoch']} | {r['phase_rms_rad']:.4f} |")
    lines+=['','## 相同 checkpoint 的光电干预','',
            '以下是正常 Top-1 减去干预 Top-1，单位为百分点。负数表示移除后反而提高；不能解释为可加和的贡献比例。','',
            '| 数据集 | 方法 | 移除全部光学 | 仅移除 Vision 光学 | 仅移除 Language 光学 | 移除电子融合输出 | 光学下降≥5 pp |',
            '|---|---|---:|---:|---:|---:|---|']
    for r in result['formal']:
        values=[r[k] for k in ('optical_removal_top1_drop_pp','vision_optical_removal_top1_drop_pp','language_optical_removal_top1_drop_pp','electronic_removal_top1_drop_pp')]
        lines.append(f"| {names[r['dataset']]} | {LABELS[r['method']]} | "+' | '.join(f'{v:+.2f}' for v in values)+(' | 通过 |' if r['optical_dependency_5pp_pass'] else ' | 未通过 |'))
    lines+=['','## 生成器是否影响 mask 与检索','',
            '| 数据集 | 方法 | 关闭 LoRA 的相位差 / rad | 关闭 LoRA 的 Top-1 下降 / pp | 换回初始专家的 Top-1 下降 / pp |',
            '|---|---|---:|---:|---:|']
    for r in result['formal']:
        if not r['method'].endswith('_lora'):continue
        a=r['ablations']
        lines.append(f"| {names[r['dataset']]} | {LABELS[r['method']]} | {r['lora_phase_effect_rad']:.5f} | {r['top1_percent']-100*a['lora_disabled_same_decoder']['top1_retrieval_accuracy']:+.2f} | {r['top1_percent']-100*a['initial_experts']['top1_retrieval_accuracy']:+.2f} |")
    lines+=['','所有“下降”统一采用正常值减去干预值：正数表示移除后退化，负数表示移除后改善。','',
            '## 开发阶段：只使用验证集','',
            '| 组 | 电子 LR | LoRA LR | 最后三轮验证 Top-1 均值 | 所选验证 Top-1 | LoRA 相位影响 / rad |',
            '|---|---:|---:|---:|---:|---:|']
    for r in result['development']['rows']:
        lines.append(f"| {r['run_id'].split('_devdet_')[-1].replace('_s42','')} | {r['electronic_lr']:g} | {r['generator_lr']:g} | {100*r['last3_validation_top1']:.2f}% | {100*r['selected_validation_top1']:.2f}% | {r['lora_phase_effect_rad']:.5f} |")
    lines+=['','frozen_e 的配置 LR 保留原值，但全部电子/读出参数冻结，实际更新 LR 为 0。encoder_focus 在第 2 轮后冻结 decoder。两组均不参与选 LR。','',
            '## 证据与限制','',
            '- 数字及 run ID：[summary.json](summary.json)，机器可读表：[metrics.csv](metrics.csv)。',
            '- 实际更新量：[parameter_updates.csv](parameter_updates.csv)，证据哈希：[evidence_manifest.json](evidence_manifest.json)。',
            '- 光学系数固定 0.6；本轮全部 RTX 4090，固定源码与严格确定性设置，未使用 A100。',
            '- 单优化 seed；Caltech 历史 warmstart 和测试暴露；理想仿真；不同编码器规模与训练参数量。不能据单次高分宣称稳定优势。','']
    (output/'完整结果.md').write_text('\n'.join(lines),encoding='utf-8')


def phase_diagnostics(runs,output):
    import torch
    from .protocol import common_anchor, physical_phase
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for dataset in ('caltech','cifar100','imagenette'):
        banks={}
        for run in runs:
            cfg=read(run/'protocol.json')
            if cfg.get('dataset',{}).get('name','caltech')!=dataset:continue
            bank=torch.load(run/'expert_bank.pt',map_location='cpu',weights_only=True)['phase_rad']
            banks[cfg['method']]=bank
            initial=physical_phase(common_anchor(cfg['seed'],dc_power=cfg['initial_expert_dc_power']))
        for left in banks:
            delta=banks[left]-initial
            piston=torch.atan2(delta.sin().mean((-2,-1)),delta.cos().mean((-2,-1)))
            centered=delta-piston[...,None,None]
            centered=torch.atan2(centered.sin(),centered.cos())
            rows.append({'dataset':dataset,'left':left,'right':'initial',
                         'piston_removed_rms_rad':float(centered.square().mean().sqrt()),
                         'per_expert_piston_removed_rms_rad':centered.square().mean((-2,-1)).sqrt().tolist()})
            for right in banks:
                if left>=right:continue
                diff=banks[left]-banks[right];diff=torch.atan2(diff.sin(),diff.cos())
                rows.append({'dataset':dataset,'left':left,'right':right,'circular_rms_rad':float(diff.square().mean().sqrt())})
        methods=('direct','qwen_lora','clip_lora')
        changes={m:torch.atan2((banks[m]-initial).sin(),(banks[m]-initial).cos()) for m in methods}
        limit=max(float(d.abs().quantile(.995)) for d in changes.values()) or 1.
        fig,axes=plt.subplots(3,8,figsize=(16,7),layout='constrained')
        for i,method in enumerate(methods):
            for j in range(8):
                im=axes[i,j].imshow(changes[method].flatten(0,1)[j],cmap='RdBu_r',vmin=-limit,vmax=limit)
                axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                if i==0:axes[i,j].set_title(('V' if j<4 else 'L')+str(j%4))
                if j==0:axes[i,j].set_ylabel(LABELS[method])
        fig.colorbar(im,ax=axes,label='Circular change from common initial phase (rad)',shrink=.8)
        fig.suptitle(dataset+' | all eight exported experts; shared 99.5% color range')
        fig.savefig(output/(dataset+'_expert_phase_changes.png'),dpi=150);plt.close(fig)
    return rows


def training_diagnostics(runs, dev_runs, output):
    """Keep raw update magnitudes separate from functional branch interventions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'electronic':'#2176ae','readout':'#70a7c8','generator_context':'#a13277',
              'generator_decoder':'#cf891c','router':'#468b39','global':'#777777','expert':'#6b50a0'}
    updates=[]
    for run in runs+dev_runs:
        for step in read(run/'steps.json'):
            for group,value in step.get('parameter_updates',{}).items():
                updates.append({'run_id':run.name,'epoch':step['epoch'],'stage':step['stage'],
                                'group':group,**value,'gradient_norm':step.get('gradient_norms',{}).get(group,0)})
    if updates:
        with (output/'parameter_updates.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(updates[0]));writer.writeheader();writer.writerows(updates)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True,layout='constrained')
    for ax,run in zip(axes.flat,dev_runs):
        subset=[r for r in updates if r['run_id']==run.name]
        for group,color in colors.items():
            selected=[r for r in subset if r['group']==group and r['relative_l2']>0]
            if not selected:continue
            ax.plot([r['epoch'] for r in selected],[r['relative_l2'] for r in selected],color=color,label=group)
        ax.set_yscale('log');ax.set_title(run.name.split('_devdet_')[-1].replace('_s42',''),fontsize=10)
        ax.set_xlabel('Epoch');ax.set_ylabel('First-step ||update|| / ||parameter||')
        for boundary in (2.5,6.5):ax.axvline(boundary,color='#cccccc',linestyle='--',lw=.7)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=9)
    fig.savefig(output/'development_parameter_updates.png',dpi=160);plt.close(fig)
    all_sets=[('development',dev_runs)] + [(d,[r for r in runs if read(r/'protocol.json').get('dataset',{}).get('name','caltech')==d]) for d in ('caltech','cifar100','imagenette')]
    for name,group_runs in all_sets:
        if not group_runs:continue
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for run in group_runs:
            cfg=read(run/'protocol.json');history=read(run/'history.json');epochs=[h['epoch'] for h in history]
            label=run.name.split('_devdet_')[-1].replace('_s42','') if name=='development' else LABELS[cfg['method']]
            axes[0].plot(epochs,[h['mean_task_loss'] for h in history],label=label)
            axes[1].plot(epochs,[100*h['live_validation']['top1_retrieval_accuracy'] for h in history])
            axes[2].plot(epochs,[h['expert_phase']['rms_change_rad'] for h in history])
        for ax,title,ylabel in zip(axes,('Training task loss','Validation Top-1','Expert phase movement'),('Loss','Accuracy (%)','Circular RMS (rad)')):
            ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel(ylabel)
            for i,h in enumerate(history):
                if i and h['stage']!=history[i-1]['stage']:
                    ax.axvline(h['epoch']-.5,color='#cccccc',linestyle='--',lw=.7)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle(name);fig.savefig(output/(name+'_training.png'),dpi=160);plt.close(fig)


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for column,dataset in enumerate(('caltech','cifar100','imagenette')):
        subset=sorted((r for r in rows if r['dataset']==dataset),key=lambda r:list(LABELS).index(r['method']))
        x=np.arange(len(subset));labels=[LABELS[r['method']] for r in subset]
        for offset,key,label,color in [(-.19,'top1_percent','Top-1','#1976b5'),(.19,'top3_percent','Top-3','#d48910')]:
            axes[0,column].bar(x+offset,[r[key] for r in subset],width=.38,label=label,color=color)
        for offset,key,label,color in [(-.19,'optical_removal_top1_drop_pp','Remove optical','#1976b5'),(.19,'electronic_removal_top1_drop_pp','Remove electronic','#ad3375')]:
            axes[1,column].bar(x+offset,[r[key] for r in subset],width=.38,label=label,color=color)
        axes[0,column].set_title(dataset+' | held-out retrieval')
        axes[0,column].set_ylim(0,105);axes[0,column].set_ylabel('Accuracy (%)')
        axes[1,column].set_ylabel('Top-1 drop (percentage points)')
        axes[1,column].axhline(5,color='#777777',linestyle=':',label='Optical dependency reference')
        axes[1,column].axhline(0,color='black',lw=.6)
        for ax in axes[:,column]:ax.set_xticks(x,labels,rotation=30,ha='right')
    axes[0,0].legend();axes[1,0].legend(fontsize=8)
    fig.savefig(output/'retrieval_and_branch_contribution.png',dpi=170);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs',nargs='*',default=[])
    parser.add_argument('--dev-runs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    summarize([Path(p) for p in args.runs],[Path(p) for p in args.dev_runs],Path(args.output))
