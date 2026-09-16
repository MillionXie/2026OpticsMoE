"""Independent prediction/selection verification and depth learning-curve export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    reference=read_csv(a.reference);checked=[];orders=[];curves={}
    for root in a.runs:
        metadata=json.loads((root/'metadata.json').read_text())
        lock=json.loads((root/'subset_test_lock.json').read_text())
        assert metadata['source_hashes']==lock['source_hashes']
        for result in json.loads((root/'results.json').read_text()):
            name=result['variant'];directory=root/'runs'/name/'seed17'
            assert hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()==lock['checkpoints'][f'runs/{name}/seed17/best.pt']
            rows=read_csv(directory/'test_predictions.csv');assert len(rows)==298
            assert len({x['sample_id'] for x in rows})==298
            y=np.array([int(x['label_true']) for x in rows]);s=np.array([float(x['score1']) for x in rows])
            assert np.bincount(y).tolist()==[229,69]
            positive=s[y==1,None];negative=s[y==0][None,:]
            auc=float(((positive>negative).sum()+.5*(positive==negative).sum())/positive.size/negative.size)
            prediction=s>.5
            assert np.array_equal(prediction,np.array([int(x['label_pred']) for x in rows]))
            confusion=[[int(((y==0)&~prediction).sum()),int(((y==0)&prediction).sum())],
                       [int(((y==1)&~prediction).sum()),int(((y==1)&prediction).sum())]]
            assert confusion==result['confusion_matrix'] and abs(auc-result['auroc'])<1e-12
            assert abs(float((prediction==y).mean())-result['accuracy'])<1e-12
            history=read_csv(directory/'history.csv');assert [int(x['epoch']) for x in history]==list(range(1,51))
            best_auc=-1.;best_mse=float('inf');best_epoch=None
            for row in history:
                va=float(row['val_auroc']);vm=float(row['val_detector_plane_mse'])
                if va>best_auc+1e-6 or (abs(va-best_auc)<=1e-6 and vm<best_mse):
                    best_auc,best_mse,best_epoch=va,vm,int(row['epoch'])
            assert best_epoch==result['selected_epoch']
            completed=json.loads((directory/'completed.json').read_text())
            assert completed['epochs']==50 and completed['updates']==7450
            assert all(completed['changed_phase_planes'].values())
            orders.append(tuple(completed['order_sha256']))
            model,depth,_,activation=name.split('_',3);depth=int(depth[1:])
            ref=next(x for x in reference if x['architecture']==model and int(x['depth'])==depth and x['activation']==activation and x['seed']=='17' and x['policy']=='fixed_0.5')
            checked.append(dict(variant=name,source_commit=metadata['git_commit'],auc=auc,accuracy=result['accuracy'],
                original_auc=float(ref['auroc']),auc_difference=auc-float(ref['auroc']),selected_epoch=best_epoch,original_selected_epoch=int(ref['selected_epoch']),
                mse_difference=result['detector_plane_mse']-float(ref['detector_plane_mse']),
                all_phase_tensors_updated=True,checkpoint_sha256=lock['checkpoints'][f'runs/{name}/seed17/best.pt']))
            curves[(model,activation,depth)]=history
    assert len(set(orders))==1
    report=dict(passed=True,models=len(checked),same_order_all_models=True,updates_per_model=7450,results=checked)
    (a.out/'independent_verification.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(11,13),layout='constrained')
    for i,(model,activation) in enumerate([('moe','relu_softsign'),('moe','off'),('d2nn','relu_softsign'),('d2nn','off')]):
        for depth in [2,4,6]:
            history=curves[(model,activation,depth)]
            epochs=[int(x['epoch']) for x in history]
            axes[i,0].plot(epochs,[float(x['train_detector_plane_mse']) for x in history],label=f'{depth} layers')
            axes[i,1].plot(epochs,[float(x['val_auroc']) for x in history],label=f'{depth} layers')
        for j in range(2):
            axes[i,j].set_title(f'{model.upper()}, OEO '+('Softsign' if activation!='off' else 'off'))
            axes[i,j].set_xlabel('Epoch');axes[i,j].grid(alpha=.2);axes[i,j].legend()
        axes[i,0].set_ylabel('Training detector-plane MSE (scaled by 100)')
        values=[float(row['val_auroc']) for depth in [2,4,6] for row in curves[(model,activation,depth)]]
        axes[i,1].set_ylabel('Validation AUROC');axes[i,1].set_ylim(min(.4,min(values)-.02),max(.85,max(values)+.02))
    fig.suptitle('Adrenal: same split, seed 17, 50 epochs per model')
    fig.savefig(a.out/'learning_curves.png',dpi=160);fig.savefig(a.out/'learning_curves.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
