"""Report optimization-seed stability separately from category expansion."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from .source_audit import audit_sources


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cohort(metrics,names):
    rows=[v for k,v in metrics['per_sku'].items() if k in names]
    n=sum(v['query_count'] for v in rows)
    if n==0: raise ValueError('Empty query cohort')
    return {'queries':n,'top1':sum(v['query_count']*v['top1_accuracy'] for v in rows)/n,
            'top3':sum(v['query_count']*v['top3_accuracy'] for v in rows)/n}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--seed-runs',nargs=6,required=True)
    parser.add_argument('--scale-runs',nargs=4,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    seeds,seed_contracts,scale_contracts,scale_rows,evidence={},{},{},[],[]
    seed_splits=set()
    source_audit=audit_sources([read(Path(p)/'final_report.json')['git_sha'] for p in args.seed_runs+args.scale_runs])
    for argument in args.seed_runs+args.scale_runs:
        run=Path(argument)
        if read(run/'status.json')['status']!='complete': raise ValueError(f'Incomplete {run}')
        r=read(run/'final_report.json'); cfg=read(run/'protocol.json')
        base_contract=json.dumps({k:v for k,v in cfg.items() if k not in {'method','seed'}},sort_keys=True)
        env=read(run/'environment.json')
        contract=(source_audit['runtime_fingerprint'],r['split_sha256'],base_contract)
        for filename in ('final_report.json','protocol.json','environment.json','history.json','transfer_manifest.json'):
            p=run/filename
            if p.exists(): evidence.append({'run_id':run.name,'file':filename,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        if argument in args.seed_runs:
            key=(cfg['seed'],r['method'])
            if key in seeds: raise ValueError(f'Duplicate {key}')
            if r['optimizer_updates']!=400 or cfg.get('training_optical_perturbations',True):
                raise ValueError('Seed stability requires the 400-step deterministic protocol')
            seeds[key]={'run_id':run.name,'top1':r['selected_live_test']['top1_retrieval_accuracy'],
                'training_git_sha':r['git_sha'],'device':env['device'],
                'final_ema_top1':r['final_ema_test']['top1_retrieval_accuracy'],'selected_epoch':r['selected_epoch'],
                'phase_rms_rad':r['selected_expert_phase']['rms_change_rad']}
            seed_contracts[key]=contract
            seed_splits.add(r['split_sha256'])
        else:
            if cfg.get('training_optical_perturbations',True) or r['counts']['test']!=600:
                raise ValueError('Expansion requires deterministic 30-category protocol')
            old=set(cfg['warmstart_categories']); new=set(cfg['selected_categories'])-old
            scale_rows.append({'method':r['method'],'run_id':run.name,
                'training_git_sha':r['git_sha'],'device':env['device'],
                'all_queries':{'queries':600,'top1':r['selected_live_test']['top1_retrieval_accuracy'],
                    'top3':r['selected_live_test']['top3_retrieval_accuracy'],'mrr':r['selected_live_test']['mrr']},
                'original_ten_queries':cohort(r['selected_live_test'],old),
                'added_twenty_queries':cohort(r['selected_live_test'],new),
                'selected_epoch':r['selected_epoch'],'phase_rms_rad':r['selected_expert_phase']['rms_change_rad'],
                'initial_experts_top1':r['ablations']['initial_experts']['top1_retrieval_accuracy'],
                'final_ema_top1':r['final_ema_test']['top1_retrieval_accuracy'],
                'fusion':r['fusion'],'lora_disabled_top1':r['ablations'].get('lora_disabled_same_decoder',{}).get('top1_retrieval_accuracy')})
            scale_contracts[r['method']]=contract
    expected={(s,m) for s in (42,43,44) for m in ('direct','qwen_lora')}
    if {r['device'] for r in seeds.values()} != {'NVIDIA GeForce RTX 3090'}:
        raise ValueError('Seed comparison requires the matched RTX 3090 reruns')
    if {r['device'] for r in scale_rows} != {'NVIDIA GeForce RTX 4090'}:
        raise ValueError('Category expansion requires completed RTX 4090 runs')
    if set(seeds)!=expected or len(seed_splits)!=1 or len(set(seed_contracts.values()))!=1:
        raise ValueError('Seed comparisons must share splits/code/config except optimization seed and method')
    if set(scale_contracts)!={'fixed','direct','qwen_frozen','qwen_lora'} or len(set(scale_contracts.values()))!=1:
        raise ValueError('Expansion methods are not matched')
    seed_rows=[{'seed':s,**{m:seeds[s,m] for m in ('direct','qwen_lora')},
                'qwen_minus_direct_pp':100*(seeds[s,'qwen_lora']['top1']-seeds[s,'direct']['top1'])} for s in (42,43,44)]
    seed_summary={m:{'mean_top1':statistics.mean(seeds[s,m]['top1'] for s in (42,43,44)),
                    'sample_std_top1':statistics.stdev(seeds[s,m]['top1'] for s in (42,43,44))} for m in ('direct','qwen_lora')}
    result={'seed_comparison':{'rows':seed_rows,'summary':seed_summary,'queries_per_seed':200,
        'note':'same 200 queries reused across seeds; do not treat them as 600 independent queries',
        'training_steps':400,'runtime_fingerprint':source_audit['runtime_fingerprint']},
        'category_expansion':{'rows':scale_rows,'training_batch_opportunities':2320,'seed':42,
        'runtime_fingerprint':source_audit['runtime_fingerprint'],
        'note':'all cohorts retrieve against the same 30-category gallery; added classes have supervised training examples'},
        'limits':['400-step seed study and 2320-batch expansion are separate experiments',
            'original-ten adaptation validation was seen by warmstart; original-ten test was previously inspected',
            'ideal optical simulation does not establish robustness to the strong hardware perturbations',
            'sample identities and augmentation settings match; random augmentation draws are not paired bit-for-bit across methods',
            'three seeds provide an initial stability check, not a broad statistical guarantee']}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (output/'source_audit.json').write_text(json.dumps(source_audit,indent=2),encoding='utf-8')
    plot(result,output)


def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels={'fixed':'Fixed','direct':'Direct','qwen_frozen':'Frozen Qwen','qwen_lora':'Qwen LoRA'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    rows=result['seed_comparison']['rows']
    for method,color in [('direct','#1976b5'),('qwen_lora','#ad3375')]:
        axes[0].plot([r['seed'] for r in rows],[100*r[method]['top1'] for r in rows],'o-',label=labels[method],color=color)
    axes[0].set_xticks([42,43,44]);axes[0].set_xlabel('Optimization seed')
    axes[0].set_ylabel('Test Top-1 (%)');axes[0].set_title('10 categories: 400 updates per seed');axes[0].legend()
    scale={r['method']:r for r in result['category_expansion']['rows']}
    methods=list(labels);x=np.arange(4)
    for offset,key,label,color in [(-.19,'original_ten_queries','Original 10 query classes','#1976b5'),(.19,'added_twenty_queries','Added 20 query classes','#d48910')]:
        axes[1].bar(x+offset,[100*scale[m][key]['top1'] for m in methods],width=.38,label=label,color=color)
    axes[1].set_xticks(x,[labels[m] for m in methods],rotation=12)
    axes[1].set_ylabel('Test Top-1 (%)');axes[1].set_title('30-category gallery: 600 test queries');axes[1].legend(fontsize=9)
    fig.savefig(output/'stability_and_expansion.png',dpi=170)
    plt.close(fig)


if __name__=='__main__': main()
