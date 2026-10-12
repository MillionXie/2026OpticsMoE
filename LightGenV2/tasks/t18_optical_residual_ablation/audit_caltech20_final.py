"""CPU-only final audit and scientific plots; read saved predictions, never infer."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
from caltech20_finalize import KEYS, read, sha, verify_arm, verify_global


def completion_barrier(out):
    assert read(out/'status.json')['state'] == 'tests_complete_awaiting_cpu_audit'
    for key in KEYS:
        result = read(out/key/'result.json')
        assert result['test_evaluations'] == 1 and not result['test_used_for_selection']
        assert result['selection_split'] == 'val' and set(result['metrics']) == {'train','val','test'}
        assert (out/key/'test_started.json').exists()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save_figure(fig, out, name):
    for ext in ['png','svg','pdf']: fig.savefig(out/(name+'.'+ext),dpi=180)
    svg=out/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Saved-artifact CPU audit only'
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    # No test labels are materialized until all eight once-only evaluations are complete.
    completion_barrier(args.run)
    lock=verify_global(args.run,args.data)
    import numpy as np
    import torch
    from butterfly_ablation import metrics
    manifest=read(args.data.parent/'data_manifest.json')
    arrays={}
    with np.load(args.data,allow_pickle=False) as cache:
        for split in ['train','val','test']:
            arrays[split]=(cache[split+'_labels'].copy(),cache[split+'_ids'].copy())
    args.out.mkdir(parents=True,exist_ok=True)
    summaries,classes,curves,receipts,confusions=[],[],[],[],[]
    selected_histories={}
    for key in KEYS:
        entry=lock['entries'][key];arm=Path(entry['arm']);fit=verify_arm(arm)
        result=read(args.run/key/'result.json');checkpoint=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        assert result['depth']==entry['depth'] and result['rho']==entry['rho'] and result['candidate']==entry['candidate']
        assert result['selected_state_kind']=='ema' and result['scope']==lock['scope']
        assert fit['epochs_completed']==100 and fit['candidate']=='base'
        assert checkpoint['config']==fit['config'] and checkpoint['sources']==fit['sources']
        assert checkpoint['data_sha256']==lock['data_sha256']
        assert sum(value.numel() for name,value in checkpoint['model'].items() if name.endswith('raw_phase'))==result['parameters']==fit['parameters']
        assert checkpoint['epoch']==result['selected_epoch']==entry['selected_epoch']
        assert checkpoint['validation']==fit['metrics']['val']
        assert result['metrics']['train']==fit['metrics']['train'] and result['metrics']['val']==fit['metrics']['val']
        assert result['checkpoint_sha256']==entry['checkpoint_sha256']
        assert result['data_sha256']==lock['data_sha256']
        assert result['global_selection_lock_sha256']==sha(args.run/'global_selection_lock.json')
        assert result['inference_source_sha256']==lock['source_sha256']
        start=read(args.run/key/'test_started.json')
        assert start['global_selection_lock_sha256']==result['global_selection_lock_sha256']
        assert start['time_unix']>=lock['lock_time_unix']
        assert sha(args.run/key/'test_predictions.csv')==result['test_predictions_sha256']
        for split in ['train','val','test']:
            labels,ids=arrays[split]
            assert ids.tolist()==manifest['split_ids'][split]
            path=(args.run/key if split=='test' else arm)/(split+'_predictions.csv')
            with path.open() as stream: rows=list(csv.DictReader(stream))
            assert [row['sample_id'] for row in rows]==ids.tolist()
            assert [int(row['label_true']) for row in rows]==labels.tolist()
            probability=np.array([[float(row[f'score{k}']) for k in range(len(manifest['classes']))] for row in rows])
            assert np.isfinite(probability).all() and np.all(probability>=0)
            assert probability.argmax(1).tolist()==[int(row['label_pred']) for row in rows]
            computed=metrics(labels,probability);saved=result['metrics'][split]
            assert computed['n']==saved['n'] and computed['support']==manifest['supports'][split]
            assert np.allclose(probability.sum(axis=1),1.,atol=1e-6,rtol=0)
            assert len(rows)==len(ids)
            for field in ['support','confusion_matrix']: assert computed[field]==saved[field]
            assert np.allclose(computed['recall'],saved['recall'],atol=1e-10,rtol=1e-10)
            for field in ['accuracy','balanced_accuracy','macro_f1','nll','balanced_nll']:
                assert np.isclose(computed[field],saved[field],atol=1e-10,rtol=1e-10)
            for true_index,true_label in enumerate(manifest['classes']):
                for predicted_index,predicted_label in enumerate(manifest['classes']):
                    confusions.append(dict(depth=entry['depth'],rho=entry['rho'],split=split,true_class=true_label,
                        predicted_class=predicted_label,count=saved['confusion_matrix'][true_index][predicted_index]))
            for index,label in enumerate(manifest['classes']):
                classes.append(dict(depth=entry['depth'],rho=entry['rho'],candidate=entry['candidate'],split=split,
                    class_name=label,support=saved['support'][index],correct=saved['confusion_matrix'][index][index],recall=saved['recall'][index]))
        history=read(arm/'history.json');selected_histories[key]=history
        window=min(10,len(history)//2)
        earlier=np.mean([row['val']['balanced_nll'] for row in history[-2*window:-window]])
        recent=np.mean([row['val']['balanced_nll'] for row in history[-window:]])
        improvement=100*(earlier-recent)/earlier
        observation='validation NLL improving at budget end; convergence unproven' if improvement>.5 and entry['selected_epoch']>len(history)-window else 'bounded fit; convergence unproven'
        summary=dict(depth=entry['depth'],rho=entry['rho'],candidate=entry['candidate'],best_epoch=entry['selected_epoch'],
            selected_candidate_epochs=fit['epochs_completed'],base_epochs=100,additional_epochs=0,fit_seconds=fit['seconds'],
            mean_epoch_seconds=float(np.mean([h['epoch_seconds'] for h in history])),
            parameters=result['parameters'],fit_observation=observation,validation_window_epochs=window,
            validation_window_improvement_percent=float(improvement),train_val_gap_pp=100*(fit['metrics']['train']['accuracy']-fit['metrics']['val']['accuracy']),
            test_evaluations=1,test_used_for_selection=False,checkpoint_sha256=entry['checkpoint_sha256'])
        for split in ['train','val','test']:
            for name,field in [('accuracy','accuracy'),('macro_recall','balanced_accuracy'),('macro_f1','macro_f1'),('nll','nll'),('balanced_nll','balanced_nll')]:
                summary[f'{split}_{name}']=result['metrics'][split][field]
            summary[f'{split}_detector_capture']=result['metrics'][split]['detector_capture']
        summaries.append(summary)
        receipts.append(dict(key=key,arm=str(arm),result_sha256=sha(args.run/key/'result.json'),
            test_started_sha256=sha(args.run/key/'test_started.json'),test_predictions_sha256=result['test_predictions_sha256'],
            selected_fit_result_sha256=sha(arm/'fit_result.json'),training_git_commit=read(arm/'metadata.json')['git_commit'],metrics=result['metrics']))
    # The registered common100 base is the only candidate; no invented test history.
    for depth,choice in lock['choices'].items():
        for candidate,info in choice['candidates'].items():
            for rho,arm in zip([0.,.3],info['arms']):
                for row in read(Path(arm)/'history.json'):
                    curves.append(dict(depth=int(depth),rho=rho,candidate=candidate,epoch=row['epoch'],
                        selected_candidate=candidate==choice['selected_candidate'],val_accuracy=row['val']['accuracy'],
                        val_balanced_nll=row['val']['balanced_nll'],train_accuracy=row.get('train',{}).get('accuracy',''),
                        train_balanced_nll=row.get('train',{}).get('balanced_nll',''),epoch_seconds=row['epoch_seconds']))
    write_csv(args.out/'final_summary.csv',summaries);write_csv(args.out/'per_class_support.csv',classes);write_csv(args.out/'confusion_matrices.csv',confusions);write_csv(args.out/'curve_data.csv',curves)
    audit=dict(cpu_checks_passed=True,scope=lock['scope'],run=str(args.run.resolve()),audit_time_unix=time.time(),
        global_selection_lock_sha256=sha(args.run/'global_selection_lock.json'),data_sha256=lock['data_sha256'],
        manifest_sha256=lock['manifest_sha256'],choices=lock['choices'],candidate_audits=lock['candidate_audits'],
        arms=receipts,summaries=summaries,total_test_evaluations=8,optical_inference_performed=False,
        audit_source_sha256=sha(Path(__file__)))
    (args.out/'final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,15),constrained_layout=True)
    for row,depth in enumerate([2,4,8,16]):
        choice=lock['choices'][str(depth)]
        for candidate,info in choice['candidates'].items():
            selected=candidate==choice['selected_candidate']
            for rho,arm,color in zip([0.,.3],info['arms'],['#2365a5','#ce6c23']):
                history=read(Path(arm)/'history.json');train=[h for h in history if 'train' in h]
                label=f'{candidate} rho={rho}'
                for ax,metric,scale in [(axes[row,0],'accuracy',100),(axes[row,1],'balanced_nll',1)]:
                    ax.plot([h['epoch'] for h in history],[scale*h['val'][metric] for h in history],
                        color=color,alpha=1 if selected else .35,linewidth=1.6 if selected else 1,linestyle='-' if selected else ':',label=label+' val')
                    ax.plot([h['epoch'] for h in train],[scale*h['train'][metric] for h in train],
                        color=color,alpha=1 if selected else .35,linestyle='--',marker='o',markersize=2,label=label+' train')
        for rho,color in zip([0.,.3],['#2365a5','#ce6c23']):
            summary=next(s for s in summaries if s['depth']==depth and s['rho']==rho)
            axes[row,0].scatter(summary['best_epoch'],100*summary['test_accuracy'],marker='*',s=85,color=color,label=f'Selected test rho={rho}')
            axes[row,1].scatter(summary['best_epoch'],summary['test_balanced_nll'],marker='*',s=85,color=color,label=f'Selected test rho={rho}')
        axes[row,0].set(ylabel='Accuracy (%)',ylim=(0,102));axes[row,1].set(ylabel='Balanced NLL')
        for ax in axes[row]:
            ax.set(title=f'L{depth}: common100',xlabel='Epoch')
            ax.grid(alpha=.15);ax.legend(fontsize=6,loc='best')
    fig.suptitle('Caltech101 count-conditioned20 subset | seed17 | common100, validation-selected EMA\nTest stars: one evaluation per selected arm; no instance/photographer independence established')
    save_figure(fig,args.out,'learning_curves_all');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    positions=np.arange(len(summaries))
    for index,(split,color) in enumerate([('train','#2365a5'),('val','#ce6c23'),('test','#278764')]):
        for ax,metric,scale in [(axes[0],'accuracy',100),(axes[1],'balanced_nll',1)]:
            ax.bar(positions+(index-1)*.25,[scale*s[f'{split}_{metric}'] for s in summaries],width=.24,label=split,color=color)
    labels=[f"L{s['depth']}\nrho={s['rho']}" for s in summaries]
    axes[0].set(ylabel='Accuracy (%)',ylim=(0,105));axes[1].set(ylabel='Balanced NLL')
    for ax in axes: ax.set_xticks(positions,labels);ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('Caltech101 count-conditioned20 subset | common100 | single seed, no error bars')
    save_figure(fig,args.out,'selected_summary');plt.close(fig)
    print(json.dumps(dict(cpu_checks_passed=True,total_test_evaluations=8,summaries=summaries)))


if __name__=='__main__': main()
