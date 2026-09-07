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
            contracts.add((result['git_sha'], result['split_sha256'], json.dumps(common,sort_keys=True)))
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
        dataset = cfg.get('dataset', {}).get('name', 'caltech')
        key = (dataset, result['method'])
        if key in contracts: raise ValueError(f'Duplicate {key}')
        comparable = {k:v for k,v in cfg.items() if k != 'method'}
        contracts[key] = (result['git_sha'], result['split_sha256'], json.dumps(comparable,sort_keys=True))
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
        for name in ('protocol.json','final_report.json','environment.json','history.json','steps.json','initialization.json','transfer_manifest.json'):
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
        flat = [{k:v for k,v in row.items() if k not in {'counts','ablations','last_routing_counts'}} for row in rows]
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        plot(rows,output)
    return result


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
