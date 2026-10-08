"""Independent CPU verification and paper figures; never selects model settings.

Reads only locked completed suites. Exports every seed and uses sample SD (ddof=1).
Local copies may replace the server's task root via --task-root.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from verify_bloodmnist import read,sha,predictions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE={
    'moe_nooeo':('MoE','#fa929d','s'),
    'moe':('MoE + OEO','#9dc7df','o'),
    'd2nn_wide_nooeo':('D2NN (full input)','#626c7a','^'),
    'd2nn_wide':('D2NN + OEO (full input)','#8272cc','D'),
    'd2nn_nooeo':('D2NN (small input)','#ac9f87','v'),
    'd2nn':('D2NN + OEO (small input)','#c6a965','P'),
}
MAIN=list(STYLE)[:4]
METRICS=['accuracy','balanced_accuracy','macro_f1','macro_ovr_auroc','balanced_nll','detector_capture']

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def csvwrite(p,rows):
    assert rows
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def export(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)
def axes_style(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out');ax.grid(axis='y',alpha=.15)
def resolve(folder,task):
    marker='/LightGenV2/demo_check/';s=str(folder).replace('\\','/')
    assert marker in s,s
    return task/s.split(marker,1)[1]

def verify_pilot(a,lock):
    reused=next(e for e in lock['entries'] if e['reused']);pilot=resolve(reused['folder'],a.task_root).parents[2];selection=read(pilot/'candidate_selection.json');metadata=read(pilot/'metadata.json');spec=metadata['specification'];means={};configs={};records=[]
    assert sha(pilot/'candidate_selection.json')==lock['pilot_selection_sha256']
    assert selection['test_read'] is False and metadata['test_read'] is False
    for candidate in spec['candidate_order']:
        entries=[e for e in selection['entries'] if e['candidate']==candidate];assert len(entries)==4 and {e['result']['arch'] for e in entries}==set(spec['architectures']);values=[]
        for e in entries:
            x=e['result'];d=resolve(e['folder'],a.task_root);m=read(d.parent/'metadata.json');assert x['seed']==17 and x['depth']==spec.get('pilot_depth',4) and m['candidate']==candidate;assert m['test_read'] is False;assert sha(d/'best_checkpoint.pt')==x['checkpoint_sha256'];assert m['sources']==lock['sources'] and m['data_sha256']==lock['data_sha256']
            if candidate in configs:assert configs[candidate]==m['config']
            else:configs[candidate]=m['config']
            for split in ['train','val']:predictions(d/(split+'_predictions.csv'),x['metrics'][split])
            h=read(d/'history.json');best=float('inf');selected=None
            for z in h:
                if z['val']['balanced_nll']<best-m['config']['min_delta']:best=z['val']['balanced_nll'];selected=z['epoch']
            assert selected==x['selected_epoch'];values.append(x['metrics']['val']['balanced_nll']);records.append(dict(candidate=candidate,arch=x['arch'],selected_epoch=selected,train_accuracy=x['metrics']['train']['accuracy'],val_accuracy=x['metrics']['val']['accuracy'],val_balanced_nll=x['metrics']['val']['balanced_nll']))
        means[candidate]=float(np.mean(values))
    chosen=min(spec['candidate_order'],key=means.get);assert means==selection['mean_validation_balanced_nll'];assert chosen==selection['chosen']==lock['candidate'];assert configs[chosen]==lock['config']
    for candidate,cfg in configs.items():
        for field,value in spec['candidates'][candidate].items():assert cfg[field]==value
    save(a.out/'pilot_verification.json',dict(passed=True,candidates=len(configs),models=len(records),selected=chosen,mean_validation_balanced_nll=means,selection_sha256=sha(pilot/'candidate_selection.json'),test_not_used_by_selection=True));csvwrite(a.out/'all_pilot_validation_results.csv',records)

def verify(a):
    root=a.run;lock=read(root/'selection_lock.json');entries=read(root/'results.json');meta=read(root/'metadata.json');cfg=lock.get('config',meta.get('config'));expected=54 if a.dataset=='bloodmnist' else 36
    assert len(entries)==expected==len(lock['entries']);assert read(root/'status.json')['state']=='complete'
    for path,digest in lock['sources'].items():assert sha(a.task_root/path)==digest,('Source mismatch',path)
    if a.dataset=='kather2016':verify_pilot(a,lock)
    assert {e['result']['name'] for e in entries}=={e['result']['name'] for e in lock['entries']}
    rows=[];histories={};preds={};audits=[];byseed={};identities=[]
    for e in entries:
        x=e['result'];name=x['name'];d=resolve(e['folder'],a.task_root);ev=root/(read(root/'evaluation_directory.json')['directory'] if (root/'evaluation_directory.json').exists() else 'evaluation')/name
        locked=next(z for z in lock['entries'] if z['result']['name']==name);assert locked=={k:e[k] for k in locked}
        training_metadata=read(d.parent/'metadata.json');assert training_metadata['config']==cfg and training_metadata['data_sha256']==lock['data_sha256'];assert all(lock['sources'][k]==v for k,v in training_metadata['sources'].items())
        assert sha(d/'best_checkpoint.pt')==x['checkpoint_sha256'];assert e['validation_replayed']
        h=read(d/'history.json');histories[name]=h;best=float('inf');chosen=None
        for z in h:
            if z['val']['balanced_nll']<best-cfg['min_delta']:best=z['val']['balanced_nll'];chosen=z['epoch']
        assert chosen==x['selected_epoch'] and h[-1]['epoch']==x['epochs_completed'];assert all(v>0 for v in x['updates'].values())
        gs=read(d/'gradients.json');assert all(np.isfinite(v) and v>=0 for z in gs for v in z['norms'].values());assert all(any(z['norms'][n]>0 for z in gs) for n in gs[0]['norms'])
        ids=[]
        for split in ['train','val']:
            ii,_,_=predictions(d/(split+'_predictions.csv'),x['metrics'][split]);ids.append(set(ii))
        ii,y,pr=predictions(ev/'test_predictions.csv',e['test']);assert not ids[0]&ids[1] and not set(ii)&(ids[0]|ids[1]);preds[name]=(ii,y,pr)
        if a.dataset=='bloodmnist':
            ci,_,_=predictions(ev/'test_clean_predictions.csv',e['clean_test']);assert ci==[ii[i] for i in read(root/'test_clean_indices.json')]
            if x['seed']==17 and x['arch'] in ['moe','d2nn']:
                with (d/'test_predictions.csv').open() as f:old=list(csv.DictReader(f))
                with (ev/'test_predictions.csv').open() as f:new=list(csv.DictReader(f))
                assert old==new,'Reused seed17 predictions changed'
        audit=read(ev/'phase_audit.json')
        for n,z in audit.items():
            assert z['classification_gradient_norm']>0 and z['changed_fraction']>0
            audits.append(dict(model=name,arch=x['arch'],depth=x['depth'],seed=x['seed'],phase=n,**{k:v for k,v in z.items() if k not in ['shape','map_key']}))
        row=dict(model=name,arch=x['arch'],depth=x['depth'],seed=x['seed'],parameters=x['parameters'],selected_epoch=x['selected_epoch'],epochs_completed=x['epochs_completed'])
        for split,metrics in [('train',x['metrics']['train']),('val',x['metrics']['val']),('test',e['test'])]:
            for metric in METRICS:row[split+'_'+metric]=metrics.get(metric,float('nan'))
        row['train_val_gap_pp']=100*(row['train_accuracy']-row['val_accuracy']);row['val_test_gap_pp']=100*(row['val_accuracy']-row['test_accuracy']);row['clean_test_accuracy']=e.get('clean_test',e['test'])['accuracy'];rows.append(row);byseed.setdefault(x['seed'],[]).append(x)
        identities.append(dict(model=name,checkpoint_sha256=x['checkpoint_sha256'],test_csv_sha256=sha(ev/'test_predictions.csv'),history_sha256=sha(d/'history.json')))
    for seed,models in byseed.items():
        for field in ['orders','transforms']:
            n=min(len(x[field]) for x in models);assert len({tuple(x[field][:n]) for x in models})==1,(seed,field)
    first=next(iter(preds.values()))
    assert all(p[0]==first[0] and np.array_equal(p[1],first[1]) for p in preds.values())
    assert set(byseed)=={17,27,37}
    for depth in [2,4,6]:
        pp=[x['parameters'] for x in rows if x['depth']==depth];assert max(pp)/min(pp)<1.02
    csvwrite(a.out/'per_seed_metrics.csv',rows);csvwrite(a.out/'phase_audit.csv',audits)
    aggregates=[]
    for arch in STYLE:
        for depth in [2,4,6]:
            group=[x for x in rows if x['arch']==arch and x['depth']==depth]
            if not group:continue
            assert len(group)==3
            for metric in ['test_'+k for k in METRICS]+['train_val_gap_pp','val_test_gap_pp','clean_test_accuracy']:
                v=[x[metric] for x in group];aggregates.append(dict(arch=arch,depth=depth,metric=metric,n=3,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),median=float(np.median(v))))
    csvwrite(a.out/'aggregate_metrics.csv',aggregates)
    pairs=[]
    comparisons=[('moe','d2nn_wide'),('moe_nooeo','d2nn_wide'),('moe_nooeo','d2nn_wide_nooeo'),('moe','moe_nooeo'),('d2nn_wide','d2nn_wide_nooeo')]
    if a.dataset=='bloodmnist':comparisons += [('d2nn_wide','d2nn'),('d2nn_wide_nooeo','d2nn_nooeo')]
    for aa,bb in comparisons:
        for d in [2,4,6]:
            vals=[]
            for seed in [17,27,37]:
                av=next(x for x in rows if (x['arch'],x['depth'],x['seed'])==(aa,d,seed));bv=next(x for x in rows if (x['arch'],x['depth'],x['seed'])==(bb,d,seed));vals.append(100*(av['test_accuracy']-bv['test_accuracy']))
            pairs.append(dict(first=aa,second=bb,depth=d,seed17_difference_pp=vals[0],seed27_difference_pp=vals[1],seed37_difference_pp=vals[2],mean_difference_pp=float(np.mean(vals)),sample_sd_pp=float(np.std(vals,ddof=1))))
    csvwrite(a.out/'paired_differences.csv',pairs)
    majority=int(np.argmax(entries[0]['result']['metrics']['train']['support']))
    report=dict(passed=True,dataset=a.dataset,models=len(rows),selection_lock_sha256=sha(root/'selection_lock.json'),results_sha256=sha(root/'results.json'),verifier_sha256=sha(Path(__file__)),selected_epochs_recomputed=True,all_prediction_metrics_recomputed=True,paired_data_order_and_transforms=True,test_ids_identical=True,weights_verified=True,identities=identities,statistics='Three training seeds on one fixed image split; sample SD, no significance claim or patient confidence interval.',training_majority_class=majority,test_majority_accuracy=float((first[1]==majority).mean()),uniform_random_expected_accuracy=.125)
    save(a.out/'independent_verification.json',report)
    return rows,histories,entries,preds

def figures(a,rows,histories,entries):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    def group(arch,d):return sorted([x for x in rows if x['arch']==arch and x['depth']==d],key=lambda z:z['seed'])
    def curve(ax,arch,key):
        label,c,m=STYLE[arch];vs=np.array([[x[key] for x in group(arch,d)] for d in [2,4,6]])*100
        ax.errorbar([2,4,6],vs.mean(1),yerr=vs.std(1,ddof=1),color=c,marker=m,lw=2,capsize=3,label=label)
        for i,d in enumerate([2,4,6]):ax.scatter(d+np.array([-.07,0,.07]),vs[i],s=16,color=c,alpha=.6)
        ax.set_xticks([2,4,6]);ax.set_xlabel('Main optical layers');axes_style(ax)
    fig,ax=plt.subplots(figsize=(7.1,4.8))
    for arch in MAIN:curve(ax,arch,'test_accuracy')
    majority=int(np.argmax(entries[0]['result']['metrics']['train']['support']));support=entries[0]['test']['support'];ax.axhline(100*support[majority]/sum(support),color='#777777',ls=':',lw=1.2,label='Training-majority baseline')
    ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(1,1));ax.set_title(f'{a.title}: mean ± sample SD (3 seeds)');export(fig,a.out,'depth_accuracy')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharey=True)
    for ax,d in zip(axes,[2,4,6]):
        for i,arch in enumerate(MAIN):
            v=np.array([x['test_accuracy'] for x in group(arch,d)])*100;c=STYLE[arch][1]
            ax.scatter(i+np.array([-.1,0,.1]),v,color=c,s=40,edgecolors='white',linewidth=.6,zorder=3);ax.plot([i-.22,i+.22],[np.median(v)]*2,'k-',lw=2);ax.scatter(i,np.mean(v),marker='D',s=70,facecolors='white',edgecolors='black',zorder=4)
        ax.set_xticks(range(4),['MoE','MoE\n+ OEO','D2NN','D2NN\n+ OEO']);ax.set_title(f'{d} main layers');axes_style(ax)
    axes[0].set_ylabel('Test accuracy (%)');axes[0].set_ylim(0,100);fig.suptitle('Each dot = one training seed; white diamond = mean; black bar = median\nD2NN uses full-aperture input');fig.tight_layout();export(fig,a.out,'seed_distribution')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    for ax,(on,off) in zip(axes,[('moe','moe_nooeo'),('d2nn_wide','d2nn_wide_nooeo')]):
        for seed in [17,27,37]:
            delta=[100*(next(x for x in group(on,d) if x['seed']==seed)['test_accuracy']-next(x for x in group(off,d) if x['seed']==seed)['test_accuracy']) for d in [2,4,6]];ax.plot([2,4,6],delta,'o-',label=f'Seed {seed}',alpha=.8)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks([2,4,6]);ax.set_xlabel('Main optical layers');ax.set_title(STYLE[off][0]);axes_style(ax);ax.legend(fontsize=8,loc='lower left')
    axes[0].set_ylabel('OEO on − off accuracy (percentage points)');fig.tight_layout();export(fig,a.out,'oeo_paired_gain')
    if a.dataset=='bloodmnist':
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
        for ax,arms,title in zip(axes,[['d2nn','d2nn_wide'],['d2nn_nooeo','d2nn_wide_nooeo']],['With OEO','Without OEO']):
            for arch in arms:curve(ax,arch,'test_accuracy')
            ax.set_title(title);ax.legend(fontsize=8)
        axes[0].set_ylabel('Test accuracy (%)');axes[0].set_ylim(0,100);fig.tight_layout();export(fig,a.out,'d2nn_input_coverage_control')
    curve_rows=[];fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,d in enumerate([2,4,6]):
        for arch in MAIN:
            c=STYLE[arch][1]
            for x in group(arch,d):
                h=histories[x['model']];axes[0,col].plot([z['epoch'] for z in h],[z['val']['accuracy']*100 for z in h],color=c,alpha=.35,lw=1)
                for z in h:
                    curve_rows.append(dict(model=x['model'],arch=arch,depth=d,seed=x['seed'],epoch=z['epoch'],train_loss=z['train_loss'],validation_accuracy=z['val']['accuracy'],validation_balanced_nll=z['val']['balanced_nll'],validation_capture=z['val']['detector_capture'],train_accuracy=z.get('train',{}).get('accuracy',''),selected=z['epoch']==x['selected_epoch']))
            vv=np.array([x['train_val_gap_pp'] for x in group(arch,d)]);j=MAIN.index(arch);axes[1,col].scatter(j+np.array([-.08,0,.08]),vv,color=c,s=30);axes[1,col].plot([j-.18,j+.18],[vv.mean()]*2,color='black',lw=1.5)
        axes[0,col].set_title(f'{d} main layers');axes[0,col].set_xlabel('Epoch');axes[0,col].set_ylim(0,100);axes[1,col].set_xticks(range(4),['MoE','MoE\n+ OEO','D2NN','D2NN\n+ OEO']);axes[1,col].axhline(0,color='black',lw=.7)
        for ax in axes[:,col]:axes_style(ax)
    axes[0,0].set_ylabel('Validation accuracy (%)');axes[1,0].set_ylabel('Selected train − validation accuracy (pp)');handles=[plt.Line2D([],[],color=STYLE[z][1],label=STYLE[z][0],lw=2) for z in MAIN];fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.5,1));fig.tight_layout(rect=[0,0,1,.90]);export(fig,a.out,'generalization_curves');csvwrite(a.out/'learning_curve_data.csv',curve_rows)
    fig,axes=plt.subplots(4,3,figsize=(13,12),sharex=True,sharey=True);seed_colors=['#3975b5','#de8b45','#348f73']
    for row,arch in enumerate(MAIN):
        for col,d in enumerate([2,4,6]):
            ax=axes[row,col]
            for x,c in zip(group(arch,d),seed_colors):
                h=histories[x['model']];tr=[z for z in h if 'train' in z];ax.plot([z['epoch'] for z in h],[100*z['val']['accuracy'] for z in h],color=c,lw=1.3);ax.plot([z['epoch'] for z in tr],[100*z['train']['accuracy'] for z in tr],color=c,ls='--',lw=1.3);ax.scatter(x['selected_epoch'],100*x['val_accuracy'],facecolors='white',edgecolors=c,s=30,zorder=4)
            ax.set_title(f'{STYLE[arch][0]}, {d} layers',fontsize=10);ax.set_ylim(0,100);axes_style(ax)
            if col==0:ax.set_ylabel('Accuracy (%)')
            if row==3:ax.set_xlabel('Epoch')
    handles=[plt.Line2D([],[],color=c,label=f'Seed {s}') for s,c in zip([17,27,37],seed_colors)]+[plt.Line2D([],[],color='black',label='Validation'),plt.Line2D([],[],color='black',ls='--',label='Training'),plt.Line2D([],[],color='black',marker='o',markerfacecolor='white',ls='',label='Selected checkpoint')]
    fig.legend(handles=handles,loc='upper center',ncol=6,fontsize=9);fig.tight_layout(rect=[0,0,1,.97]);export(fig,a.out,'paired_train_validation_curves')
    fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained');conf=[]
    for ax,arch in zip(axes,MAIN):
        cms=[np.array(e['test']['confusion_matrix']) for e in entries if e['result']['arch']==arch and e['result']['depth']==6];cm=np.mean([c/c.sum(1,keepdims=True) for c in cms],axis=0);im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set_title(STYLE[arch][0]);ax.set_xticks(range(8));ax.set_yticks(range(8));ax.set_xlabel('Predicted class')
        for i in range(8):
            for j in range(8):ax.text(j,i,f'{100*cm[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if cm[i,j]>.5 else 'black');conf.append(dict(arch=arch,depth=6,true_class=i,predicted_class=j,mean_fraction=cm[i,j]))
    axes[0].set_ylabel('True class');fig.colorbar(im,ax=axes,shrink=.65,label='Mean within-class fraction');fig.suptitle('6 main layers: confusion matrices averaged over 3 seeds');export(fig,a.out,'confusion_L6');csvwrite(a.out/'confusion_L6.csv',conf)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--task-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',choices=['bloodmnist','kather2016'],required=True);p.add_argument('--refresh-figures',action='store_true');a=p.parse_args();a.title={'bloodmnist':'BloodMNIST','kather2016':'Kather2016'}[a.dataset]
    previous=None
    if a.refresh_figures:
        previous=read(a.out/'independent_verification.json');assert previous['passed'] and previous['dataset']==a.dataset;assert previous['results_sha256']==sha(a.run/'results.json') and previous['selection_lock_sha256']==sha(a.run/'selection_lock.json'),'A layout refresh cannot replace underlying results'
    else:a.out.mkdir(parents=True,exist_ok=False)
    rows,histories,entries,_=verify(a);figures(a,rows,histories,entries)
    if previous is not None:
        history=read(a.out/'render_history.json') if (a.out/'render_history.json').exists() else [];history.append(dict(previous_verifier_sha256=previous['verifier_sha256'],current_verifier_sha256=sha(Path(__file__)),unchanged_results_sha256=previous['results_sha256'],all_checks_repeated=True));save(a.out/'render_history.json',history)
    print(json.dumps(dict(verified=len(rows),output=str(a.out))))
if __name__=='__main__':main()
