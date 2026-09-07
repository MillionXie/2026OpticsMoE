"""Compare completed staged runs and plot physical phases on shared scales."""
import argparse
import hashlib
import json
from pathlib import Path


LABELS = {'fixed':'Fixed experts','direct':'Direct phase optimization',
          'qwen_frozen':'Frozen Qwen + decoder','qwen_lora':'Qwen LoRA + decoder'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(runs, output):
    import torch
    from .protocol import common_anchor,physical_phase
    output.mkdir(parents=True,exist_ok=True)
    rows, evidence, histories, contracts, banks = [], [], {}, [], {}
    for run in runs:
        if read(run/'status.json')['status'] != 'complete':
            raise ValueError(f'Incomplete run: {run}')
        result = read(run/'final_report.json')
        cfg = read(run/'protocol.json')
        history = read(run/'history.json')
        architecture = read(run/'architecture.json')
        phase_bank = torch.load(run/'expert_bank.pt',map_location='cpu',weights_only=True)['phase_rad']
        delta = phase_bank-physical_phase(common_anchor(cfg['seed'],dc_power=cfg['initial_expert_dc_power']))
        piston = torch.atan2(delta.sin().mean((-2,-1)),delta.cos().mean((-2,-1)))
        centered = delta-piston[...,None,None]
        centered = torch.atan2(centered.sin(),centered.cos())
        update_rows = centered.flatten(0,1).flatten(1)
        update_rows = update_rows-update_rows.mean(1,keepdim=True)
        if result['selected_expert_phase']['rms_change_rad'] > 1e-6 and float(update_rows.norm(dim=1).min()) > 1e-8:
            normalized = torch.nn.functional.normalize(update_rows,dim=1)
            correlations = normalized@normalized.T
            update_correlation = float(correlations[~torch.eye(8,dtype=torch.bool)].mean())
        else:
            update_correlation = None
        method = result['method']
        banks[method] = phase_bank
        comparable_cfg = {k:v for k,v in cfg.items() if k!='method'}
        contracts.append((result['git_sha'],result['split_sha256'],json.dumps(comparable_cfg,sort_keys=True)))
        histories[method] = history
        rows.append({'method':method,'label':LABELS[method],'run_id':run.name,
            'device':read(run/'environment.json')['device'],
            'selected_epoch':result['selected_epoch'],'selected_live_test':result['selected_live_test'],
            'final_ema_test':result['final_ema_test'],'selected_validation':history[result['selected_epoch']-1]['live_validation'],
            'selected_phase_rms_rad':result['selected_expert_phase']['rms_change_rad'],
            'selected_phase_piston_removed_rms_rad':float(centered.square().mean().sqrt()),
            'mean_cross_expert_update_correlation':update_correlation,
            'last_phase_rms_rad':history[-1]['expert_phase']['rms_change_rad'],
            'fusion':result['fusion'],'ablations':result['ablations'],'counts':result['counts'],
            'optimizer_updates':result['optimizer_updates'],'batch_opportunities':result['training_batch_opportunities'],
            'initial_validation':read(run/'initial_validation.json'),
            'stage_ends':[{k:h[k] for k in ('epoch','stage','mean_task_loss','live_validation',
                'stage_end_validation_with_initial_experts','expert_phase','expert_selection_counts','task_gradient_chain')} for h in history if h['stage_end_validation_with_initial_experts'] is not None],
            'export_max_error':result['export_max_error'],
            'generator_trainable':architecture['generator_trainable'],
            'peak_memory_gib':result['peak_memory_gib'],'elapsed_seconds':result['elapsed_seconds']})
        for p in sorted(run.iterdir()):
            if p.suffix in {'.json','.yaml'}:
                evidence.append({'run_id':run.name,'file':p.name,'bytes':p.stat().st_size,
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if len(set(contracts))!=1:
        raise ValueError('Runs have different code, splits, or training protocols')
    methods = {r['method'] for r in rows}
    if len(rows)!=len(methods) or methods not in (set(LABELS),{'direct','qwen_lora'}):
        raise ValueError('Expected all four methods or the direct/Qwen-LoRA primary pair')
    report = {'protocol':'staged_alpha40','training_git_sha':contracts[0][0],'split_sha256':contracts[0][1],
        'resolved_protocol':comparable_cfg,
        'rows':rows,'selection':'live adaptation-validation Top-1 then MRR; no test selection',
        'limitations':['one optimization seed per comparison','original-ten validation was seen by the historical warmstart',
            'original-ten test was exposed in the earlier pilot diagnosis','different parameter counts and computation costs'],
        'time_boundary':'training, checkpoint I/O and evaluation; excludes model loading and source hashing; CUDA simulation only'}
    if len({row['device'] for row in rows}) > 1:
        report['limitations'].append('GPU models differ; this is not a matched-device performance or timing comparison')
    phase_pairs = []
    for i, left in enumerate(sorted(banks)):
        for right in sorted(banks)[i+1:]:
            delta = banks[left]-banks[right]
            delta = torch.atan2(delta.sin(),delta.cos())
            piston = torch.atan2(delta.sin().mean((-2,-1)),delta.cos().mean((-2,-1)))
            centered = delta-piston[...,None,None]
            centered = torch.atan2(centered.sin(),centered.cos())
            phase_pairs.append({'left':left,'right':right,'circular_rms_rad':float(delta.square().mean().sqrt()),
                'piston_removed_rms_rad':float(centered.square().mean().sqrt()),
                'per_modality_rms_rad':delta.square().mean((1,2,3)).sqrt().tolist()})
    report['cross_method_selected_phase'] = phase_pairs
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    plot(histories,runs,output)
    return report


def plot(histories,runs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from .protocol import common_anchor, physical_phase
    colors = {'fixed':'#777777','direct':'#1976b5','qwen_frozen':'#d48910','qwen_lora':'#ad3375'}
    fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for method,history in histories.items():
        epochs = [r['epoch'] for r in history]
        color = colors[method]
        axes[0,0].plot(epochs,[r['mean_task_loss'] for r in history],label=LABELS[method],color=color)
        axes[0,1].plot(epochs,[100*r['live_validation']['top1_retrieval_accuracy'] for r in history],color=color)
        axes[1,0].plot(epochs,[r['expert_phase']['rms_change_rad'] for r in history],color=color)
        axes[1,1].plot(epochs,[min(a for row in r['fusion'].values() for a in row) for r in history],color=color)
    names = [('Training task loss','Loss'),('Live adaptation-validation Top-1','Accuracy (%)'),
             ('Expert movement from common initialization','Circular phase RMS (rad)'),('Minimum of four optical fusion coefficients','Coefficient')]
    for ax,(title,ylabel) in zip(axes.flat,names):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        boundaries = [r['epoch']-.5 for i,r in enumerate(history) if i and r['stage']!=history[i-1]['stage']]
        for stage_boundary in boundaries: ax.axvline(stage_boundary,color='#bbbbbb',lw=.8,linestyle='--')
    axes[1,1].axhline(.4,color='#444444',linestyle=':',lw=1,label='Required floor')
    axes[1,1].set_ylim(.39,.56)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(output/'training_comparison.png',dpi=160); plt.close(fig)
    banks = {}
    for run in runs:
        cfg = read(run/'protocol.json')
        bank = torch.load(run/'expert_bank.pt',map_location='cpu',weights_only=True)['phase_rad']
        initial = physical_phase(common_anchor(cfg['seed'],dc_power=cfg['initial_expert_dc_power']))
        delta = bank-initial
        banks[cfg['method']] = torch.atan2(delta.sin(),delta.cos())
    methods = [m for m in LABELS if m in banks]
    fig,axes = plt.subplots(2,len(methods),figsize=(3*len(methods),6),layout='constrained')
    limit = max(float(d.abs().quantile(.995)) for d in banks.values()) or 1.
    for c,method in enumerate(methods):
        for m,modality in enumerate(('Vision','Language')):
            im = axes[m,c].imshow(banks[method][m,0],cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[m,c].set_title(LABELS[method],fontsize=10)
            axes[m,c].set_xticks([]); axes[m,c].set_yticks([])
            if c==0: axes[m,c].set_ylabel(f'{modality} expert 0')
    fig.colorbar(im,ax=axes,label='Circular phase change (rad)',shrink=.8)
    fig.suptitle('Validation-selected expert phase changes (shared scale; 99.5% range)')
    fig.savefig(output/'selected_phase_changes.png',dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    summarize([Path(p) for p in args.runs],Path(args.output))
