"""CPU-only screened20 final audit/plots; saved probabilities, no inference."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import caltech_selected_finalize as f
from audit_caltech20_final import write_csv, save_figure


def completion_barrier(run):
    assert f.read(run/'status.json')['state']=='tests_complete_awaiting_cpu_audit'
    assert f.read(run/'current_processes.json')['active']==[]
    for key in f.KEYS:
        result=f.read(run/key/'result.json')
        assert result['test_evaluations']==1 and not result['test_used_for_selection']
        assert result['selection_split']=='val' and set(result['metrics'])=={'train','val','test'}
        assert (run/key/'test_started.json').exists()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU saved-artifact audit only'
    parser=argparse.ArgumentParser()
    for key in ('run','data','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    completion_barrier(a.run);lock=f.verify_global(a.run,a.data)
    import numpy as np
    import torch
    from butterfly_ablation import metrics
    manifest=f.read(a.data.parent/'data_manifest.json')
    with np.load(a.data,allow_pickle=False) as z:
        arrays={s:(z[s+'_labels'].copy(),z[s+'_ids'].copy()) for s in ('train','val','test')}
    a.out.mkdir(parents=True,exist_ok=True)
    summaries,supports,confusions,curves,receipts=[],[],[],[],[]
    histories={}
    for key in f.KEYS:
        entry=lock['entries'][key];arm=Path(entry['arm']);fit=f.verify_arm(arm)
        result=f.read(a.run/key/'result.json')
        best=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        assert result['depth']==entry['depth'] and result['rho']==entry['rho'] and result['candidate']=='base'
        assert result['scope']==lock['scope'] and result['selected_state_kind']=='ema'
        assert result['checkpoint_sha256']==entry['checkpoint_sha256'] and result['data_sha256']==lock['data_sha256']
        assert result['global_selection_lock_sha256']==f.sha(a.run/'global_selection_lock.json')
        assert result['inference_source_sha256']==lock['source_sha256']
        assert best['epoch']==entry['selected_epoch']==result['selected_epoch']
        assert best['config']==last['config']==fit['config'] and best['sources']==last['sources']==fit['sources']
        assert best['data_sha256']==last['data_sha256']==lock['data_sha256'] and last['epoch']==100
        assert result['parameters']==fit['parameters']==sum(v.numel() for k,v in best['model'].items() if k.endswith('raw_phase'))
        assert last['initial_parameter_sha256']==fit['initial_parameter_sha256']
        assert last['orders']==fit['orders'] and last['transforms']==fit['transforms']
        assert best['validation']==fit['metrics']['val']
        assert all(result['metrics'][s]==fit['metrics'][s] for s in ('train','val'))
        start=f.read(a.run/key/'test_started.json')
        assert start['global_selection_lock_sha256']==result['global_selection_lock_sha256']
        assert start['time_unix']>=lock['lock_time_unix']
        assert f.sha(a.run/key/'test_predictions.csv')==result['test_predictions_sha256']
        for split in ('train','val','test'):
            labels,ids=arrays[split]
            assert ids.tolist()==manifest['split_ids'][split]
            path=(a.run/key if split=='test' else arm)/(split+'_predictions.csv')
            with path.open() as stream:rows=list(csv.DictReader(stream))
            assert [r['sample_id'] for r in rows]==ids.tolist()
            assert [int(r['label_true']) for r in rows]==labels.tolist()
            p=np.array([[float(r[f'score{k}']) for k in range(20)] for r in rows])
            assert np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1,atol=1e-6,rtol=0)
            assert p.argmax(1).tolist()==[int(r['label_pred']) for r in rows]
            actual=metrics(labels,p);saved=result['metrics'][split]
            assert actual['n']==saved['n']==sum(manifest['supports'][split])
            assert actual['support']==saved['support']==manifest['supports'][split]
            assert actual['confusion_matrix']==saved['confusion_matrix']
            for field in ('accuracy','balanced_accuracy','macro_f1','nll','balanced_nll'):
                assert np.isclose(actual[field],saved[field],atol=1e-10,rtol=1e-10)
            assert np.allclose(actual['recall'],saved['recall'],atol=1e-10,rtol=1e-10)
            for i,c in enumerate(manifest['classes']):
                supports.append(dict(depth=entry['depth'],rho=entry['rho'],split=split,class_name=c,
                    support=saved['support'][i],correct=saved['confusion_matrix'][i][i],recall=saved['recall'][i]))
                for j,other in enumerate(manifest['classes']):
                    confusions.append(dict(depth=entry['depth'],rho=entry['rho'],split=split,true_class=c,predicted_class=other,count=saved['confusion_matrix'][i][j]))
        history=f.read(arm/'history.json');histories[key]=history
        early=np.mean([h['val']['balanced_nll'] for h in history[-20:-10]])
        recent=np.mean([h['val']['balanced_nll'] for h in history[-10:]])
        improvement=100*(early-recent)/early
        train_points=[h for h in history if 'train' in h]
        before=next(h for h in train_points if h['epoch']==90)
        train_improvement=100*(before['train']['balanced_nll']-history[-1]['train']['balanced_nll'])/before['train']['balanced_nll']
        observation=('validation still improving; budget limited, convergence unproven' if improvement>.5 else
            'validation plateau/worsening despite training improvement; generalization concern' if train_improvement>1 else
            'bounded fit; convergence unproven')
        summary=dict(depth=entry['depth'],rho=entry['rho'],candidate='base',best_epoch=entry['selected_epoch'],
            base_epochs=100,additional_epochs=0,selected_candidate_epochs=100,parameters=fit['parameters'],fit_seconds=fit['seconds'],
            mean_epoch_seconds=float(np.mean([h['epoch_seconds'] for h in history])),fit_observation=observation,
            validation_window_improvement_percent=float(improvement),train_last10_nll_improvement_percent=float(train_improvement),
            train_val_gap_pp=100*(fit['metrics']['train']['accuracy']-fit['metrics']['val']['accuracy']),
            train_val_macro_gap_pp=100*(fit['metrics']['train']['balanced_accuracy']-fit['metrics']['val']['balanced_accuracy']),
            test_evaluations=1,test_used_for_selection=False,checkpoint_sha256=entry['checkpoint_sha256'])
        for split in ('train','val','test'):
            for name,field in [('accuracy','accuracy'),('macro_recall','balanced_accuracy'),('macro_f1','macro_f1'),('nll','nll'),('balanced_nll','balanced_nll')]:
                summary[f'{split}_{name}']=result['metrics'][split][field]
            summary[f'{split}_detector_capture']=result['metrics'][split]['detector_capture']
        summaries.append(summary)
        receipts.append(dict(key=key,arm=str(arm),training_git_commit=f.read(arm/'metadata.json')['git_commit'],
            result_sha256=f.sha(a.run/key/'result.json'),test_started_sha256=f.sha(a.run/key/'test_started.json'),
            test_predictions_sha256=result['test_predictions_sha256'],metrics=result['metrics']))
        for h in history:
            row=dict(depth=entry['depth'],rho=entry['rho'],candidate='base',epoch=h['epoch'],epoch_seconds=h['epoch_seconds'],mixup_sha256=h['mixup_sha256'])
            for split in ('train','val'):
                for name,field in [('accuracy','accuracy'),('macro_recall','balanced_accuracy'),('balanced_nll','balanced_nll')]:
                    row[f'{split}_{name}']=h.get(split,{}).get(field,'')
            curves.append(row)
    for filename,rows in [('final_summary.csv',summaries),('per_class_support.csv',supports),('confusion_matrices.csv',confusions),('curve_data.csv',curves)]:
        write_csv(a.out/filename,rows)
    audit=dict(cpu_checks_passed=True,scope=lock['scope'],run=str(a.run.resolve()),audit_time_unix=time.time(),
        global_selection_lock_sha256=f.sha(a.run/'global_selection_lock.json'),data_sha256=lock['data_sha256'],
        manifest_sha256=lock['manifest_sha256'],classes=manifest['classes'],supports=manifest['supports'],choices=lock['choices'],
        arms=receipts,summaries=summaries,total_test_evaluations=8,optical_inference_performed=False,
        audit_source_sha256=f.sha(Path(__file__)),plot_helper_sha256=f.sha(Path(__file__).with_name('audit_caltech20_final.py')))
    (a.out/'final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(f.DEPTHS),3,figsize=(16,15),constrained_layout=True)
    for row,depth in enumerate(f.DEPTHS):
        for rho,color,name in [(0.,'#2365a5','rho0'),(.3,'#ce6c23','rho03')]:
            history=histories[f'L{depth}_{name}'];train=[h for h in history if 'train' in h]
            summary=next(s for s in summaries if s['depth']==depth and s['rho']==rho)
            for ax,metric,field,scale in [(axes[row,0],'accuracy','accuracy',100),(axes[row,1],'balanced_accuracy','macro_recall',100),(axes[row,2],'balanced_nll','balanced_nll',1)]:
                ax.plot([h['epoch'] for h in history],[scale*h['val'][metric] for h in history],color=color,label=f'rho={rho} val')
                ax.plot([h['epoch'] for h in train],[scale*h['train'][metric] for h in train],color=color,linestyle='--',label=f'rho={rho} train')
                ax.scatter(summary['best_epoch'],scale*summary['test_'+field],marker='*',s=80,color=color,label=f'rho={rho} once test')
        for col,label in enumerate(('Accuracy (%)','Macro recall (%)','Balanced NLL')):
            axes[row,col].set(title=f'L{depth}, common100',xlabel='Epoch',ylabel=label)
            if col<2:axes[row,col].set_ylim(0,102)
            axes[row,col].grid(alpha=.15);axes[row,col].legend(fontsize=6)
    fig.suptitle('Caltech101 validation-screened20 | regularized | seed17 | exploratory development\nDashed train, solid validation, one test star per selected EMA; no error bars')
    save_figure(fig,a.out,'learning_curves_all');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    positions=np.arange(8)
    for j,(split,color) in enumerate([('train','#2365a5'),('val','#ce6c23'),('test','#278764')]):
        for ax,field,scale in zip(axes,('accuracy','macro_recall','balanced_nll'),(100,100,1)):
            ax.bar(positions+(j-1)*.25,[scale*s[split+'_'+field] for s in summaries],width=.24,label=split,color=color)
    for ax,label in zip(axes,('Accuracy (%)','Macro recall (%)','Balanced NLL')):
        ax.set_xticks(positions,[f"L{s['depth']}\nrho={s['rho']}" for s in summaries]);ax.set_ylabel(label);ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('Selected EMA | screened20 development | class supports unequal | single seed')
    save_figure(fig,a.out,'selected_summary');plt.close(fig)
    for depth in f.DEPTHS:
        fig,axes=plt.subplots(1,2,figsize=(18,9),constrained_layout=True)
        for ax,rho,name in zip(axes,(0.,.3),('rho0','rho03')):
            cm=np.array(f.read(a.run/f'L{depth}_{name}'/'result.json')['metrics']['test']['confusion_matrix'])
            colors=cm/np.array(manifest['supports']['test'])[:,None]
            im=ax.imshow(colors,vmin=0,vmax=1,cmap='Blues')
            ax.set_xticks(range(20),manifest['classes'],rotation=90,fontsize=7);ax.set_yticks(range(20),manifest['classes'],fontsize=7)
            for i in range(20):
                for j in range(20):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if colors[i,j]>.55 else 'black')
            ax.set(title=f'L{depth} rho={rho}: once test, counts / row recall colors',xlabel='Predicted',ylabel='True')
            fig.colorbar(im,ax=ax,label='Within-class fraction',shrink=.65)
        save_figure(fig,a.out,f'test_confusion_L{depth}');plt.close(fig)
    print(json.dumps(dict(cpu_checks_passed=True,total_test_evaluations=8,summaries=summaries)))


if __name__=='__main__':main()
