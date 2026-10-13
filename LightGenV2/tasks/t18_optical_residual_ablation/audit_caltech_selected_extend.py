"""CPU-only common200 continuation and endpoint audit; saved artifacts only."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import caltech_selected_extend as e


def check_predictions(path, split, data, manifest, saved):
    with e.np.load(data,allow_pickle=False) as z:
        labels,ids=z[split+'_labels'].copy(),z[split+'_ids'].copy()
    with path.open() as stream:rows=list(csv.DictReader(stream))
    assert [r['sample_id'] for r in rows]==ids.tolist()==manifest['split_ids'][split]
    assert [int(r['label_true']) for r in rows]==labels.tolist()
    p=e.np.array([[float(r[f'score{k}']) for k in range(20)] for r in rows])
    assert e.np.isfinite(p).all() and (p>=0).all() and e.np.allclose(p.sum(1),1,atol=1e-6,rtol=0)
    assert p.argmax(1).tolist()==[int(r['label_pred']) for r in rows]
    actual=e.o.metrics(labels,p)
    assert actual['n']==sum(manifest['supports'][split])==saved['n']
    for key in ('support','confusion_matrix'):assert actual[key]==saved[key]
    assert actual['support']==manifest['supports'][split]
    for key in ('accuracy','balanced_accuracy','macro_f1','nll','balanced_nll'):
        assert e.np.isclose(actual[key],saved[key],atol=1e-10,rtol=1e-10)
    assert e.np.allclose(actual['recall'],saved['recall'],atol=1e-10,rtol=1e-10)


def verify_arm(arm,data,require_complete=True):
    meta=e.read(arm/'metadata.json');parent=Path(meta['parent'])
    previous,prior,manifest=e.verify_parent(parent,data)
    cfg=e.read(arm/'config.json');assert cfg==e.extension_config(previous['config'])
    sources=e.source_identity(parent);assert meta['sources']==sources
    e.verify_sources(sources,meta['git_commit'])
    history=e.read(arm/'history.json');last=e.torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    best=e.torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    count=len(history)
    assert 100<count<=200 and (count==200 or not require_complete)
    assert history[:100]==e.read(parent/'history.json')
    assert [r['epoch'] for r in history]==list(range(1,count+1))
    selected=min(history,key=lambda h:h['val']['balanced_nll'])
    assert best['epoch']==selected['epoch'] and best['validation']==selected['val']
    assert last['epoch']==count and last['scheduler']['last_epoch']==count
    assert best['config']==last['config']==cfg and best['sources']==last['sources']==sources
    assert best['data_sha256']==last['data_sha256']==manifest['cache_sha256']
    assert last['depth']==best['depth']==previous['depth']
    assert last['initial_parameter_sha256']==previous['initial_parameter_sha256']
    assert len(last['orders'])==len(last['transforms'])==count
    assert last['orders'][:100]==previous['orders'] and last['transforms'][:100]==previous['transforms']
    samples=sum(manifest['supports']['train']);steps=count*((samples+15)//16)
    assert len(last['optimizer']['state'])==1+5*previous['depth']
    assert all(float(v['step'])==steps for v in last['optimizer']['state'].values())
    for state in (last['model'],last['ema'],best['model']):
        assert all(bool(e.torch.isfinite(v).all()) for v in state.values())
    gradients=e.read(arm/'gradients.json')
    assert gradients[:100]==e.read(parent/'gradients.json') and [g['epoch'] for g in gradients]==list(range(1,count+1))
    assert all(len(g['norms'])==1+5*previous['depth'] and all(e.np.isfinite(v) and v>0 for v in g['norms'].values()) for g in gradients)
    receipt=e.read(arm/'resume_receipt.json')
    assert receipt['parent_last_checkpoint_sha256']==e.sha(parent/'last_checkpoint.pt')
    assert receipt['parent_best_checkpoint_sha256']==e.sha(parent/'best_checkpoint.pt')
    assert receipt['restored_state_sha256']=={k:e.state_digest(prior[k]) for k in ('model','ema','optimizer','scheduler')}
    assert receipt['parent_epoch']==100 and receipt['first_new_epoch']==101 and receipt['total_epochs']==200
    assert not any(receipt[k] for k in ('optimizer_reset','scheduler_reset','ema_reset'))
    assert receipt['restored_lr']==prior['optimizer']['param_groups'][0]['lr'] and receipt['restored_scheduler_last_epoch']==100
    smoke=e.read(arm/'smoke.json')
    assert smoke['passed'] and smoke['depth']==previous['depth'] and smoke['rho']==previous['rho']
    assert smoke['microbatch']==cfg['microbatch'] and smoke['expert_batched']
    assert smoke['restored_raw_parameter_sha256']==receipt['restored_state_sha256']['model']
    assert smoke['probability_max_error']<=1e-6 and smoke['gradient_max_relative_l2']<=.02 and smoke['gradient_min_cosine']>=.999 and smoke['gradients_finite_nonzero']
    for row in history:
        epoch=row['epoch'];mix=e.parent_tools.expected_mixup_hash(cfg['seed'],epoch,samples,cfg['mixup_alpha'])
        assert row['mixup_sha256']==mix
        if epoch>100:
            order=e.r.epoch_order(cfg['seed'],epoch,samples)
            theta=e.o.b.affine_parameters(samples,cfg['seed'],epoch,cfg['augmentation'])
            assert last['orders'][epoch-1]==e.r.sha_tensor(order)
            assert last['transforms'][epoch-1]==e.hashlib.sha256((e.r.sha_tensor(theta)+mix).encode()).hexdigest()
    parameters=sum(v.numel() for k,v in best['model'].items() if k.endswith('raw_phase'))
    assert parameters==previous['parameters']==10000+439848*(previous['depth']//2)
    check_predictions(arm/'val_predictions.csv','val',data,manifest,best['validation'])
    if require_complete:
        fit=e.read(arm/'fit_result.json')
        assert fit['epochs_completed']==200 and fit['stop_reason']=='maximum_epochs' and fit['test_evaluations']==0
        assert fit['config']==cfg and fit['sources']==sources and fit['data_sha256']==manifest['cache_sha256']
        assert fit['parameters']==parameters and fit['selected_epoch']==best['epoch'] and fit['metrics']['val']==best['validation']
        assert fit['checkpoint_sha256']==e.sha(arm/'best_checkpoint.pt') and fit['history_sha256']==e.sha(arm/'history.json')
        assert fit['selected_model_state_sha256']==e.state_digest(best['model'])
        assert fit['orders']==last['orders'] and fit['transforms']==last['transforms']
        assert fit['initial_parameter_sha256']==previous['initial_parameter_sha256']
        assert not fit['test_used_for_selection'] and fit['selection_split']=='val' and set(fit['metrics'])=={'train','val'}
        check_predictions(arm/'train_predictions.csv','train',data,manifest,fit['metrics']['train'])
        if best['epoch']<=100:
            old=e.torch.load(parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            assert e.state_digest(best['model'])==e.state_digest(old['model']) and fit['metrics']==previous['metrics']
        return fit
    return dict(cpu_checks_passed=True,epochs=count,depth=previous['depth'],rho=previous['rho'],selected_epoch=best['epoch'],
        source_count=len(sources),resume_verified=True,phase_count=1+5*previous['depth'],mixup_hashes_verified=count)


def verify_pair(fits,depth):
    assert [f['depth'] for f in fits]==[depth,depth] and [f['rho'] for f in fits]==[0.,.3]
    a,b=[f['config'] for f in fits]
    assert {k for k in set(a)|set(b) if a.get(k)!=b.get(k)}=={'residual_rho'}
    for key in ('sources','orders','transforms','parameters','initial_parameter_sha256','data_sha256'):
        assert fits[0][key]==fits[1][key],key


def verify_all(base,data):
    status=e.read(base/'status.json');meta=e.read(base/'metadata.json')
    assert status['state']=='fit_complete_awaiting_validation_review' and not status['failures']
    assert e.read(base/'current_processes.json')==dict(active=[],pending=[])
    assert not Path('/proc',str(meta['pid'])).exists()
    assert set(e.read(base/'fit_manifest.json')['arms'])==set(e.KEYS)
    for job in status['completed']:assert job['exit_code']==0 and not Path('/proc',str(job['pid'])).exists()
    fits={key:verify_arm(base/key,data) for key in e.KEYS}
    for depth in e.DEPTHS:verify_pair([fits[f'L{depth}_{r}'] for r in ('rho0','rho03')],depth)
    assert all(f['sources']==meta['sources'] for f in fits.values())
    return fits


def endpoint_barrier(final):
    status=e.read(final/'status.json')
    assert status['state']=='tests_complete_awaiting_cpu_audit' and not status['failures']
    assert e.read(final/'current_processes.json')==dict(active=[],pending=[])
    assert not Path('/proc',str(e.read(final/'suite_started.json')['pid'])).exists()
    for key in e.KEYS:
        result=e.read(final/key/'result.json');claim=e.read(final/key/'test_started.json')
        assert not Path('/proc',str(claim['pid'])).exists()
        assert result['test_evaluations'] in (0,1) and set(result['metrics'])=={'train','val','test'}
        assert not result['test_used_for_selection'] and result['selection_split']=='val'


def report(base,data,out,final=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    fits=verify_all(base,data);manifest=e.parent_tools.verify_data(data);out.mkdir(parents=True,exist_ok=True)
    summaries=[];curves=[];support_rows=[];cm_rows=[];histories={};endpoints={}
    if final is not None:
        endpoint_barrier(final)
        from caltech_selected_extend_finalize import verify_global
        verify_global(final,data)
        status=e.read(final/'status.json');assert status['state']=='tests_complete_awaiting_cpu_audit' and not status['failures']
        assert e.read(final/'current_processes.json')==dict(active=[],pending=[])
        assert not Path('/proc',str(e.read(final/'suite_started.json')['pid'])).exists()
        for key in e.KEYS:
            result=e.read(final/key/'result.json');claim=e.read(final/key/'test_started.json')
            assert not Path('/proc',str(claim['pid'])).exists()
            assert not result['test_used_for_selection'] and result['selection_split']=='val'
            assert result['test_evaluations'] in (0,1) and result['metrics']['train']==fits[key]['metrics']['train']
            assert result['metrics']['val']==fits[key]['metrics']['val'] and result['selected_epoch']==fits[key]['selected_epoch']
            assert result['selected_model_state_sha256']==fits[key]['selected_model_state_sha256']
            assert result['global_selection_lock_sha256']==e.sha(final/'global_selection_lock.json')
            assert claim['global_selection_lock_sha256']==result['global_selection_lock_sha256']
            assert result['checkpoint_sha256']==fits[key]['checkpoint_sha256']
            assert result['test_predictions_sha256']==e.sha(final/key/'test_predictions.csv')
            check_predictions(final/key/'test_predictions.csv','test',data,manifest,result['metrics']['test'])
            parent=Path(e.read(base/key/'metadata.json')['parent'])
            old_best=e.torch.load(parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            same_parameters=e.state_digest(old_best['model'])==fits[key]['selected_model_state_sha256']
            assert result['test_reused']==same_parameters and result['test_evaluations']==int(not same_parameters)
            if result['test_evaluations']==0:
                parent=Path(e.read(base/key/'metadata.json')['parent']);old=e.read(parent.parent/'final'/key/'result.json')
                assert result['reused_parent_test_result_sha256']==e.sha(parent.parent/'final'/key/'result.json')
                assert result['metrics']['test']==old['metrics']['test'] and result['test_predictions_sha256']==old['test_predictions_sha256']
                assert result['test_reused'] and claim['policy']=='Reuse saved parent test; no inference'
            else:assert not result['test_reused'] and claim['time_unix']>=e.read(final/'global_selection_lock.json')['lock_time_unix']
            endpoints[key]=result
    for key,fit in fits.items():
        history=e.read(base/key/'history.json');histories[key]=history
        row=dict(depth=fit['depth'],rho=fit['rho'],best_epoch=fit['selected_epoch'],epochs=200,additional_epochs=100,
            continuation_seconds=fit['continuation_seconds'],total_fit_seconds=fit['seconds'],parameters=fit['parameters'],
            train_val_gap_pp=100*(fit['metrics']['train']['accuracy']-fit['metrics']['val']['accuracy']),
            late_val_nll_improvement_percent=100*(e.np.mean([h['val']['balanced_nll'] for h in history[-20:-10]])-e.np.mean([h['val']['balanced_nll'] for h in history[-10:]]))/e.np.mean([h['val']['balanced_nll'] for h in history[-20:-10]]),
            checkpoint_sha256=fit['checkpoint_sha256'],selected_model_state_sha256=fit['selected_model_state_sha256'])
        all_metrics=endpoints[key]['metrics'] if final else fit['metrics']
        for split,metric in all_metrics.items():
            for name,field in [('accuracy','accuracy'),('macro_recall','balanced_accuracy'),('macro_f1','macro_f1'),('balanced_nll','balanced_nll'),('nll','nll')]:row[split+'_'+name]=metric[field]
            for i,c in enumerate(manifest['classes']):
                support_rows.append(dict(depth=fit['depth'],rho=fit['rho'],split=split,class_name=c,support=metric['support'][i],recall=metric['recall'][i]))
                for j,other in enumerate(manifest['classes']):cm_rows.append(dict(depth=fit['depth'],rho=fit['rho'],split=split,true_class=c,predicted_class=other,count=metric['confusion_matrix'][i][j]))
        if final:row.update(test_evaluations=endpoints[key]['test_evaluations'],test_reused=endpoints[key]['test_reused'])
        summaries.append(row)
        for h in history:
            entry=dict(depth=fit['depth'],rho=fit['rho'],epoch=h['epoch'],epoch_seconds=h['epoch_seconds'])
            for split in ('train','val'):
                for name,field in [('accuracy','accuracy'),('macro_recall','balanced_accuracy'),('balanced_nll','balanced_nll')]:entry[split+'_'+name]=h.get(split,{}).get(field,'')
            curves.append(entry)
    from audit_caltech20_final import write_csv,save_figure
    for name,rows in [('summary.csv',summaries),('curve_data.csv',curves),('per_class_support.csv',support_rows),('confusion_matrices.csv',cm_rows)]:write_csv(out/name,rows)
    audit=dict(cpu_checks_passed=True,optical_inference_performed=False,scope=e.SCOPE,time_unix=time.time(),classes=manifest['classes'],supports=manifest['supports'],
        summaries=summaries,arms={key:dict(fit_result_sha256=e.sha(base/key/'fit_result.json'),parent=e.read(base/key/'metadata.json')['parent']) for key in e.KEYS},
        endpoint_count=len(endpoints),new_test_evaluations=sum(r['test_evaluations'] for r in endpoints.values()),reused_parent_tests=sum(r['test_reused'] for r in endpoints.values()))
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(16,15),constrained_layout=True)
    for i,depth in enumerate(e.DEPTHS):
        for rho,name,color in ((0.,'rho0','#2365a5'),(.3,'rho03','#ce6c23')):
            key=f'L{depth}_{name}';history=histories[key];train=[h for h in history if 'train' in h];summary=next(s for s in summaries if s['depth']==depth and s['rho']==rho)
            for j,(field,column,scale) in enumerate((('accuracy','accuracy',100),('balanced_accuracy','macro_recall',100),('balanced_nll','balanced_nll',1))):
                ax=axes[i,j];ax.plot([h['epoch'] for h in history],[scale*h['val'][field] for h in history],color=color,label=f'rho={rho} val')
                ax.plot([h['epoch'] for h in train],[scale*h['train'][field] for h in train],color=color,linestyle='--',label=f'rho={rho} train')
                if final:ax.scatter(summary['best_epoch'],scale*summary['test_'+column],color=color,marker='*',s=90,label=f'rho={rho} selected test'+(' reused' if summary['test_reused'] else ' once'))
                parent=Path(e.read(base/key/'metadata.json')['parent']);old=e.read(parent.parent/'final'/key/'result.json')
                ax.scatter(old['selected_epoch'],scale*old['metrics']['test'][field],color=color,marker='o',facecolors='none',s=35,label=f'rho={rho} previous100 test')
        for j,label in enumerate(('Accuracy (%)','Macro recall (%)','Balanced NLL')):
            axes[i,j].set(title=f'L{depth} common200',xlabel='Epoch',ylabel=label);axes[i,j].axvline(100,color='#888888',alpha=.4,linestyle=':');axes[i,j].grid(alpha=.15);axes[i,j].legend(fontsize=5)
            if j<2:axes[i,j].set_ylim(0,102)
    fig.suptitle('Screened Caltech20 | seed17 | exact-state continuation100 to200 | exploratory development\nDashed train, solid val; hollow previous100 test, star current selected endpoint; no seed error bars')
    save_figure(fig,out,'learning_curves_all');plt.close(fig)
    if final:
        for depth in e.DEPTHS:
            fig,axes=plt.subplots(1,2,figsize=(18,9),constrained_layout=True)
            for ax,rho,name in zip(axes,(0.,.3),('rho0','rho03')):
                cm=e.np.array(endpoints[f'L{depth}_{name}']['metrics']['test']['confusion_matrix']);colors=cm/e.np.array(manifest['supports']['test'])[:,None]
                im=ax.imshow(colors,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(20),manifest['classes'],rotation=90,fontsize=7);ax.set_yticks(range(20),manifest['classes'],fontsize=7)
                for i in range(20):
                    for j in range(20):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if colors[i,j]>.55 else 'black')
                ax.set(title=f'L{depth} rho={rho}: selected endpoint, counts / row recall colors',xlabel='Predicted',ylabel='True');fig.colorbar(im,ax=ax,shrink=.65)
            save_figure(fig,out,f'test_confusion_L{depth}');plt.close(fig)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','data','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--final',type=Path)
    a=p.parse_args();print(json.dumps(report(a.base,a.data,a.out,a.final)))
