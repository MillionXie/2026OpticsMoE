"""One fixed selected checkpoint per model; evaluate both A and B test domains."""
import csv,json,time
import numpy as np
import torch
from eurosat_data import *
from eurosat_runtime import *

def seal():
    verify_source();models={}
    for model in ('moe','A_only','B_only','AB'):
        folders=[MOE/s for s in STAGES] if model=='moe' else [D2/model/s for s in STAGES] if model=='AB' else [D2/model/model]
        finals=[json.loads((p/'final.json').read_text()) for p in folders]
        assert all(x['state']=='complete' and not x['test_evaluated'] for x in finals)
        cp=paths(model);state=checkpoint(cp)
        models[model]=dict(path=str(cp),sha256=file_sha(cp),selected_stage=state['stage'],selected_epoch=state['epoch'],validation=state['validation'],
            total_steps=sum(x['total_steps'] for x in finals),total_epoch_seconds=sum(x['total_epoch_seconds'] for x in finals),
            domain_presentations={d:sum(x['domain_presentations'][d] for x in finals) for d in ('A','B')})
    assert models['moe']['total_steps']==models['AB']['total_steps']==17500
    assert models['moe']['domain_presentations']==models['AB']['domain_presentations']=={'A':525000,'B':525000}
    assert models['A_only']['total_steps']==8750 and models['A_only']['domain_presentations']['B']==0
    assert models['B_only']['total_steps']==8750 and models['B_only']['domain_presentations']['A']==0
    value=dict(models=models,source_sha256=verify_source(),split_sha256=PLAN['split_records_sha256'],sealed_at=time.time(),policy='Four models sealed before test; one checkpoint per model for both domains')
    dest=ROOT/'runs/SELECTION_SEAL.json'
    if dest.exists():
        previous=json.loads(dest.read_text());assert previous['models']==models
    else:atomic(dest,value)

def run():
    seal();selected=json.loads((ROOT/'runs/SELECTION_SEAL.json').read_text());results={};dest=ROOT/'results';dest.mkdir(exist_ok=True)
    for model,info in selected['models'].items():
        architecture='moe' if model=='moe' else 'd2nn';loaded,s,r,h=build(architecture,dest/model)
        try:
            state=checkpoint(info['path']);assert file_sha(info['path'])==info['sha256'];M.restore(r,h,state)
            before=C.state_digest(M.clone(r,h));entry=dict(checkpoint=info,tests={})
            for domain in ('A','B'):
                file=dest/model/f'{domain}_test.npz';metrics=evaluate(loaded,r,h,s,domain,'test',dest=file)
                # Recompute all key values from persisted logits and labels, independently of loop totals.
                with np.load(file) as z:
                    y=z['labels'];pred=z['logits'].argmax(1);assert np.array_equal(pred,z['predictions'])
                    expected=[i for i,rec in enumerate(RECORDS) if rec['domain']==domain and rec['split']=='test']
                    assert np.array_equal(z['indices'],expected)
                    assert np.array_equal(y,[RECORDS[i]['label'] for i in expected])
                    check=E.metrics_from_predictions(y,pred,metrics['loss'],metrics['elapsed_sec'])
                    for key in ('accuracy','macro_f1','samples','confusion_matrix'):assert check[key]==metrics[key]
                entry['tests'][domain]=metrics
            assert before==C.state_digest(M.clone(r,h));entry['same_fixed_checkpoint_both_domains']=True
            entry['mean_accuracy']=sum(entry['tests'][d]['accuracy'] for d in ('A','B'))/2
            entry['worst_accuracy']=min(entry['tests'][d]['accuracy'] for d in ('A','B'))
            if model=='moe':
                entry['diagnostics']={}
                for mode in ('uniform','isolated'):
                    entry['diagnostics'][mode]={d:evaluate(loaded,r,h,s,d,'test',mode,dest/model/f'diagnostic_{mode}_{d}.npz') for d in ('A','B')}
                initial=checkpoint(ROOT/'runs/initial_moe.pt')
                for group in ('A','B'):
                    M.restore(r,h,state)
                    for name,p in M.named(r,h).items():
                        n=C.expert_number(name)
                        if n is not None and (n<2)==(group=='A'):
                            prefix,key=name.split('.',1)
                            with torch.no_grad():p.copy_(initial[prefix][key].to(p.device))
                    entry['diagnostics'][f'restore_{group}_initial_phases']={d:evaluate(loaded,r,h,s,d,'test',dest=dest/model/f'diagnostic_reset_{group}_{d}.npz') for d in ('A','B')}
                M.restore(r,h,state)
            atomic(dest/model/'metrics.json',entry);results[model]=entry
        finally:r.close()
    atomic(dest/'PERFORMANCE.json',dict(state='complete',seed=42,models=results,split_sha256=PLAN['split_records_sha256'],test_based_selection=False))
    table=[]
    for model,entry in results.items():
        a,b=entry['tests']['A'],entry['tests']['B']
        table.append(dict(model=model,A_accuracy=a['accuracy'],B_accuracy=b['accuracy'],A_macro_f1=a['macro_f1'],B_macro_f1=b['macro_f1'],mean_accuracy=entry['mean_accuracy'],worst_accuracy=entry['worst_accuracy']))
    with (dest/'performance.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    render(results,table,dest)
    atomic(ROOT/'runs/queue_status.json',dict(state='complete',completed_at=time.time(),performance=str(dest/'PERFORMANCE.json')))

def render(results,table,dest):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    names={'moe':'MoE A+B','A_only':'D2NN A-only','B_only':'D2NN B-only','AB':'D2NN A+B'}
    for row in table:ax.plot(['Optical A','SAR B'],[100*row['A_accuracy'],100*row['B_accuracy']],marker='o',label=names[row['model']])
    ax.set(ylabel='Test accuracy (%)',ylim=(0,100),title='EuroSAT optical/SAR: fixed checkpoint evaluated on both domains (seed 42)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(dest/'domain_performance.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(11,19))
    for row,(model,entry) in enumerate(results.items()):
        for col,d in enumerate(('A','B')):
            ax=axs[row,col];ax.imshow(entry['tests'][d]['confusion_matrix'],cmap='Blues');ax.set(title=names[model]+' / '+d,xlabel='Predicted category',ylabel='True category',xticks=range(10),yticks=range(10))
    fig.tight_layout();fig.savefig(dest/'confusion_matrices.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,(model,entry) in zip(axs.flat,results.items()):
        folders=[MOE/s for s in STAGES] if model=='moe' else [D2/model/s for s in STAGES] if model=='AB' else [D2/model/model]
        history=[json.loads(p.read_text()) for folder in folders for p in sorted(folder.glob('epoch_*.json'))]
        for d in ('A','B'):ax.plot(range(1,len(history)+1),[100*v['validation'][d]['accuracy'] if d in v['validation'] else np.nan for v in history],label=d)
        ax.set(title=names[model],xlabel='Curriculum epoch',ylabel='Validation accuracy (%)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(dest/'validation_curves.png',dpi=180);plt.close(fig)
    lines=['# EuroSAT optical/SAR performance, seed 42','','One fixed checkpoint per model, evaluated on both domains. A/B labels never used for normal automatic inference.','','| Model | A accuracy | B accuracy | A macro-F1 | B macro-F1 | Mean accuracy |','|---|---:|---:|---:|---:|---:|']
    for row in table:lines.append('| '+names[row['model']]+' | '+' | '.join(f'{row[k]*100:.2f}%' for k in ('A_accuracy','B_accuracy','A_macro_f1','B_macro_f1','mean_accuracy'))+' |')
    lines+=['','A: Sentinel-2 optical RGB; B: Sentinel-1 VV/VH encoded into three channels. Same ten classes and paired locations. Geographic groups and nearby patches are kept in one partition; exact split and counts are in PLAN/SPLIT/DATA_CHECKS. Single seed 42.','','MoE and D2NN A+B have identical sampled indices, augmentation seeds, and 17,500 optimizer steps, verified per epoch. Their losses, optical parameter counts, and teacher computation differ. A-only and B-only select only on their source validation domain. Diagnostic isolated routing uses domain IDs and is not the normal MoE result.','']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':run()
