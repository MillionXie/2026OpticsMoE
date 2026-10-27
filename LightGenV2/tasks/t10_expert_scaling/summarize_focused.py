"""Independently audit downloaded focused-training artifacts; never run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(snapshot, report):
    manifest=read(snapshot/'transfer_manifest.json')
    for name,entry in manifest.items():
        f=snapshot/name
        assert f.stat().st_size==entry['size'] and digest(f)==entry['sha256']
    server=read(snapshot/'final_independent_server_audit.json')
    assert server['release_verified'] and server['all_own_pids_exited']
    selection=read(snapshot/'recipe_selection.json')
    rows=[]; histories={}; routes={}
    for name in ['color_s17','phase_color_s17','distill_s17']:
        directory=snapshot/name;result=read(directory/'result.json');history=read(directory/'history.json');metadata=read(directory/'metadata.json')
        assert result['focused_rounds']==len(history) and result['optimized_train_images']==1024*len(history)
        assert len(history)>=10 and (history[-1]['patience']>=6 or len(history)==20)
        assert all(g['finite'] and g['norm']>0 for e in history for g in e['gradients'].values())
        assert result['checkpoint_sha256']==server['methods'][name]['best_sha256']
        for split,n in [('train',3496),('val',752)]:
            cm=np.asarray(result[split]['confusion_matrix']);assert cm.sum()==n and abs(cm.trace()/n-result[split]['accuracy'])<1e-7
        with np.load(directory/'initial_train_predictions.npz',allow_pickle=False) as z:
            train_ids=z['ids'];labels=z['labels'];assert len(labels)==3496 and np.isfinite(z['scores']).all()
        chosen=read(directory/'current_train_selection.json');indices=np.asarray(chosen['train_indices'])
        assert len(np.unique(indices))==1024 and np.bincount(labels[indices],minlength=8).tolist()==[128]*8
        assert train_ids[indices].tolist()==chosen['ids']
        with np.load(directory/'latest_val_predictions.npz',allow_pickle=False) as z:
            scores=z['scores'];y=z['labels'];assert len(y)==752 and np.isfinite(scores).all()
            cm=np.zeros((8,8),int);np.add.at(cm,(y,scores.argmax(1)),1)
            assert np.array_equal(cm,history[-1]['val']['confusion_matrix'])
        route={}
        for split,n in [('train',3496),('val',752)]:
            with np.load(directory/f'selected_{split}_routes.npz',allow_pickle=False) as z:
                rp=z['route_probabilities'].astype(float);mask=z['route_mask'];assert rp.shape==(n,49) and np.isfinite(rp).all()
                assert np.allclose(rp.sum(1),1,atol=1e-6) and np.all(mask==1)
                neff=np.exp(-(rp*np.log(np.maximum(rp,1e-30))).sum(1));top24=np.sort(rp,axis=1)[:,-24:].sum(1)
                route[split]=dict(samples=n,hard_mask_all_one=True,hard_mask_interpretation='Definition of full-k; not evidence of balanced specialization',
                    effective_experts_definition='exp(entropy) per sample',participation_ratio_mean=float((1/(rp*rp).sum(1)).mean()),effective_experts_mean=float(neff.mean()),effective_experts_median=float(np.median(neff)),effective_experts_min=float(neff.min()),
                    top24_power_mean=float(top24.mean()),mean_soft_min=float(rp.mean(0).min()),mean_soft_max=float(rp.mean(0).max()))
        routes[name]=route
        previous=0.;durations=[]
        for e in history:durations.append(e['elapsed_seconds']-previous);previous=e['elapsed_seconds']
        rows.append(dict(recipe=result['recipe'],seed=17,completed_focused_rounds=len(history),optimized_train_images=result['optimized_train_images'],
            source_full_adaptation_epochs=result['source_adaptation_epochs'],total_full_epoch_equivalent=result['total_full_epoch_equivalent'],
            selected_logical_counter=result['best_epoch'],selected_focused_round=result['best_epoch']-metadata['continuation_source_lock']['source_epoch'],
            selected_kind=result['best_kind'],train_full_accuracy=result['train']['accuracy'],validation_accuracy=result['val']['accuracy'],
            validation_nll=result['val']['macro_nll'],test_accuracy=None,test_not_selected=True,
            checkpoint_sha256=result['checkpoint_sha256'],last_checkpoint_sha256=server['methods'][name]['last_sha256'],
            actual_round_wall_median_seconds=float(np.median(durations[1:])),
            round_wall_includes_refresh_and_swa=True,stage_history_elapsed_seconds=history[-1]['elapsed_seconds'],
            gradients_finite_nonzero=True,route=route))
        histories[name]=history
    selected=min(rows,key=lambda r:(-r['validation_accuracy'],r['validation_nll']))
    assert selected['recipe']==selection['selected']['recipe']=='distill'
    test=read(snapshot/'distill_s17/test_result.json');predictions=snapshot/'distill_s17/test_predictions.npz'
    assert digest(predictions)==test['predictions_sha256'] and test['checkpoint_sha256']==selected['checkpoint_sha256']
    with np.load(predictions,allow_pickle=False) as z:
        ids=z['ids'];y=z['labels'];scores=z['scores'];prediction=z['predictions']
        assert len(y)==752 and np.isfinite(scores).all() and np.array_equal(scores.argmax(1),prediction)
        cm=np.zeros((8,8),int);np.add.at(cm,(y,prediction),1);correct=int((y==prediction).sum())
        assert np.array_equal(cm,test['test']['confusion_matrix']) and abs(correct/752-test['test']['accuracy'])<1e-7
    original=report.parent/'kather_n49k49_continue90to120_20261027/final_snapshot/N49_k49_s17'
    old_test=read(original/'test_result.json');assert digest(original/'test_predictions.npz')==old_test['predictions_sha256']
    with np.load(original/'test_predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels']);old_prediction=z['predictions']
    with np.load(original/'selected_val_routes.npz',allow_pickle=False) as z:
        old_route=z['route_probabilities'].astype(float);old_route_ids=z['ids']
    with np.load(snapshot/'distill_s17/selected_val_routes.npz',allow_pickle=False) as z:
        new_route=z['route_probabilities'].astype(float);assert np.array_equal(old_route_ids,z['ids'])
    route_comparison=dict(split='VAL',mean_absolute_probability_change=float(np.abs(new_route-old_route).mean()),
        old_entropy_neff_mean=float(np.exp(-(old_route*np.log(old_route)).sum(1)).mean()),
        new_entropy_neff_mean=float(np.exp(-(new_route*np.log(new_route)).sum(1)).mean()),
        old_participation_ratio_mean=float((1/(old_route*old_route).sum(1)).mean()),
        new_participation_ratio_mean=float((1/(new_route*new_route).sum(1)).mean()),
        interpretation='Near-uniform full-k soft power remains; no demonstrated increase in expert specialization.')
    old_wrong=old_prediction!=y;new_wrong=prediction!=y
    old_cm=np.zeros((8,8),int);np.add.at(old_cm,(y,old_prediction),1)
    selected.update(test_accuracy=correct/752,test_correct=correct,test_samples=752,test_not_selected=False)
    paired=dict(reference='Same seed17 original120 VAL-selected model',old_correct=int((~old_wrong).sum()),new_correct=correct,
        old_errors_corrected=int((old_wrong&~new_wrong).sum()),old_correct_now_wrong=int((~old_wrong&new_wrong).sum()),
        both_wrong=int((old_wrong&new_wrong).sum()),delta_percentage_points=(correct/752-(~old_wrong).mean())*100,
        old_per_class_correct=old_cm.diagonal().tolist(),new_per_class_correct=cm.diagonal().tolist())
    baseline=read(report.parent/'expert_tail_followup_20261026/final_summary_20261027.json')
    refs=[r for r in baseline['rows'] if r['experts']==49 and r['seed']==17]
    comparison=[dict(version='Original90 N49/k'+str(r['top_k']),seed=17,test_correct=r['test_correct'],test_accuracy=r['test_accuracy']) for r in refs]
    comparison.append(dict(version='Original120 N49/k49',seed=17,test_correct=paired['old_correct'],test_accuracy=paired['old_correct']/752))
    comparison.append(dict(version='Focused development selected distill',seed=17,test_correct=correct,test_accuracy=correct/752))
    summary=dict(role='Single-seed extra-budget post-TEST method development; target not achieved',dataset='Kather2016',classes=baseline['classes'],
        samples=dict(train=3496,val=752,test=752),source_commit=server['source_commit'],host_head=server['host_head'],
        validation_only_recipe_selection=True,only_selected_recipe_tested=True,all_three_methods_complete=True,rows=rows,selected_recipe='distill',
        independent_test_audit=dict(correct=correct,total=752,accuracy=correct/752,confusion_matrix=cm.tolist(),argmax_matches=True,finite_scores=True,
            predictions_sha256=digest(predictions),cross_reference_ids_labels_equal=True),paired_test_changes=paired,same_seed_comparison=comparison,selected_route_comparison=route_comparison,
        outcome='Selected method TEST is5/752 lower than original120 and9/752 lower than same-seed90 k24. No robust improvement established; do not expand seeds or budgets.',
        cost=dict(focused_rounds_sum=sum(r['completed_focused_rounds'] for r in rows),optimized_train_images_sum=sum(r['optimized_train_images'] for r in rows),
            earlier_short_and_extended_costs_preserved_separately=True,rounds_are_not_full3496_image_epochs=True,
            runtime_note='First round includes initial fullTRAIN scoring. Later walltime includes periodic TRAIN refresh, circular SWA validation, checkpoint writes. Different samples per round are not equal-work kernel speedups.'),
        provenance=dict(transfer_manifest_sha256=digest(snapshot/'transfer_manifest.json'),entries=len(manifest),all_transferred_sha_verified=True,
            server_weight_data_sha_audit=server,source_test_result_sha256=digest(original/'test_result.json')),
        release=dict(all_own_pids_exited=True,own_pids=server['own_pids'],matching_active_processes=server['matching_active_processes'],gpu_status=server['gpu_status']),
        limitations=['Only seed17; no three-seed mean or SD.',
            'TEST errors informed development; not an independent confirmatory test or equal-budget Top-k ablation.',
            'Probe TRAIN metrics use512 fixed images; final train_full_accuracy uses3496.',
            'Latest VAL scores belong to final training round, not necessarily selected checkpoint.',
            'Current selection and initial/latest refreshed TRAIN scores are archived; intermediate replaced difficulty caches and per-round selection lists were not all retained, so complete historical replay coverage is not retrospectively proven.',
            'Failed temperature trials and earlier short trials remain preserved; no discarded costs or overwritten original best/last.'])
    report.mkdir(parents=True,exist_ok=True);(report/'final_summary_20261027.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8));names={'color_s17':'Color','phase_color_s17':'Color + phase noise','distill_s17':'Color + gated KD'}
    for name,h in histories.items():
        rounds=[e['focused_round'] for e in h];line=axes[0,0].plot(rounds,[e['val']['accuracy']*100 for e in h],label=names[name])[0]
        axes[0,1].plot(rounds,[e['val']['macro_nll'] for e in h],color=line.get_color(),label=names[name])
        swa=[e for e in h if e.get('circular_swa_val')]
        axes[0,0].plot([e['focused_round'] for e in swa],[e['circular_swa_val']['accuracy']*100 for e in swa],ls='--',color=line.get_color(),alpha=.6)
        axes[1,0].plot(rounds,[e['elapsed_seconds']-(h[i-1]['elapsed_seconds'] if i else 0) for i,e in enumerate(h)],label=names[name],color=line.get_color())
    axes[0,0].axhline(78.324467,color='k',ls=':',label='Inherited best')
    axes[0,1].axhline(.621844411,color='k',ls=':')
    axes[0,0].set(title='VAL accuracy: EMA solid, circular SWA dashed',ylabel='Accuracy (%)',xlabel='Focused round (1024 TRAIN samples)')
    axes[0,1].set(title='VAL macro-NLL',ylabel='NLL',xlabel='Focused round')
    axes[1,0].set(title='Actual round walltime, including refresh / SWA',ylabel='Seconds',xlabel='Focused round')
    x=np.arange(8);axes[1,1].bar(x-.18,old_cm.diagonal()/94*100,.36,label='Original120')
    axes[1,1].bar(x+.18,cm.diagonal()/94*100,.36,label='Selected focused KD')
    axes[1,1].set_xticks(x,['TUM','STR','CMP','LYM','DEB','MUC','ADI','EMP']);axes[1,1].set(title='Observed TEST per-class recall',ylabel='Recall (%)',ylim=(50,100))
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8);axes[1,1].legend(fontsize=8)
    fig.suptitle('N49/k49 seed17: focused post-TEST development, no net TEST gain')
    fig.tight_layout();fig.savefig(report/'final_convergence_20261027.png',dpi=160);plt.close(fig)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    s=summarize(a.snapshot,a.report);print(json.dumps(dict(selected=s['selected_recipe'],test=s['independent_test_audit'],paired=s['paired_test_changes'],rows=[{k:r[k] for k in ['recipe','completed_focused_rounds','train_full_accuracy','validation_accuracy','test_accuracy']} for r in s['rows']]),ensure_ascii=False))
