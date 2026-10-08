"""Deterministic successes AND errors from locked 6-layer seed17 predictions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_oeo_suite import read,sha,save,csvwrite,export,MAIN
from plot_input_examples import CLASSES

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',choices=['bloodmnist','kather2016'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert read(a.run/'status.json')['state']=='complete';assert sha(a.data)==read(a.run/'metadata.json')['data_sha256'];a.out.mkdir(parents=True,exist_ok=False)
    results=read(a.run/'results.json');preds={}
    for arch in MAIN:
        e=next(e for e in results if (e['result']['arch'],e['result']['depth'],e['result']['seed'])==(arch,6,17));p=a.run/(read(a.run/'evaluation_directory.json')['directory'] if (a.run/'evaluation_directory.json').exists() else 'evaluation')/e['result']['name']/'test_predictions.csv'
        with p.open() as f:preds[arch]={r['sample_id']:r for r in csv.DictReader(f)}
    with np.load(a.data,allow_pickle=False) as z:
        x=z['test_images'];y=z['test_labels'].reshape(-1);ids=z['test_ids'] if 'test_ids' in z else np.array([f'test_{i}' for i in range(len(y))])
    names=CLASSES[a.dataset];index={sid:i for i,sid in enumerate(ids)};chosen=[];fig,axes=plt.subplots(2,8,figsize=(17,7.7));short=['M','M+OEO','D','D+OEO'];plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    for k in range(8):
        groups=[[],[]]
        for sid in sorted(ids[y==k]):
            assert all(int(preds[arch][sid]['label_true'])==k for arch in MAIN);correct=all(int(preds[arch][sid]['label_pred'])==k for arch in MAIN);groups[0 if correct else 1].append(sid)
        for row in range(2):
            ax=axes[row,k];ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'{k}: {names[k]}',fontsize=9)
            if not groups[row]:ax.text(.5,.5,'No example',ha='center',va='center');ax.set_frame_on(False);continue
            sid=groups[row][0];ax.imshow(x[index[sid]],interpolation='nearest');labels=[str(sid)];record=dict(sample_id=str(sid),true_class=k,group='all_four_correct' if row==0 else 'at_least_one_wrong',depth=6,seed=17)
            for abbr,arch in zip(short,MAIN):
                r=preds[arch][sid];pr=int(r['label_pred']);score=float(r[f'score{pr}']);labels.append(f'{abbr}: {pr} ({score:.2f})'+(' *' if pr!=k else ''));record[arch+'_prediction']=pr;record[arch+'_score']=score
            chosen.append(record);ax.set_xlabel('\n'.join(labels),fontsize=8)
    axes[0,0].set_ylabel('All four correct',fontsize=11);axes[1,0].set_ylabel('At least one wrong',fontsize=11);fig.suptitle(f'{a.dataset}: fixed 6-layer seed17 examples\nFirst sample ID per class and outcome; * = incorrect. M = MoE; D = full-input D2NN. Scores are not calibrated confidence.',fontsize=11);fig.tight_layout();export(fig,a.out,'prediction_examples');csvwrite(a.out/'prediction_examples.csv',chosen);save(a.out/'selection_manifest.json',dict(rule='Lexicographically first test ID within each true class and each of two outcomes: all four arms correct, or at least one wrong. No substitution when empty.',depth=6,seed=17,data_sha256=sha(a.data),results_sha256=sha(a.run/'results.json'),script_sha256=sha(Path(__file__)),n_examples=len(chosen)))
if __name__=='__main__':main()
