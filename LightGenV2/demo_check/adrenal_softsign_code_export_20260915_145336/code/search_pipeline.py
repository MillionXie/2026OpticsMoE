"""Preregistered validation screening, paired formal runs, then locked test."""
import argparse,json,traceback
import numpy as np
import torch
from experiments import *
from run_experiment import (setup,snapshot,read,save,sha,status,now,getdata,train_one,evaluate,csvwrite)
from models import build
from calibration import choose_thresholds,full_metrics

def validation_predictions(v,seed,val):
    dest=run_dir(v,seed);ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
    assert sha(dest/'best.pt')==read(dest/'selection.json')['sha256']
    model=build(v['architecture'],CONFIGS[v['id']]).cuda();model.load_state_dict(ck['model'],strict=True)
    vm,rows=evaluate(model,val,True)
    assert abs(vm['auroc']-ck['validation']['auroc'])<1e-12
    csvwrite(dest/'validation_predictions.csv',rows)
    y=np.array([r['label_true'] for r in rows]);p=np.array([[r['score0'],r['score1']] for r in rows])
    record=choose_thresholds(y,p);record.update(checkpoint_sha256=sha(dest/'best.pt'),validation_auroc=vm['auroc'],
      validation_predictions_sha256=sha(dest/'validation_predictions.csv'))
    save(dest/'thresholds.json',record)
    del model,ck;torch.cuda.empty_cache()

def run(resume=False):
    setup();source=snapshot();lock=ROOT/'protocol/training_lock.json'
    assert read(ROOT/'protocol/preflight.json')['source_hashes']==source
    if lock.exists():
        assert resume and read(lock)['source_hashes']==source
    else:save(lock,{'created_at':now(),'source_hashes':source,'search':SEARCH,'experiment':EXP})
    train=getdata('train');val=getdata('val');assert not set(train[2])&set(val[2])
    screen=[v for v in VARIANTS if v['depth']==SEARCH['screen_depth'] and v['oeo']]
    index=0
    for seed in SEARCH['screen_seeds']:
        for v in screen:
            index+=1;train_one(v,seed,train,val,index,source)
    # Only validation metrics enter this selection. Same shared activation for both architectures.
    rows=[]
    for act in SEARCH['candidates']:
        mm=[read(run_dir(v,s)/'selection.json')['validation'] for v in screen if v['activation']==act for s in SEARCH['screen_seeds']]
        assert len(mm)==4
        rows.append({'activation':act,'n':4,**{k:float(np.mean([m[k] for m in mm])) for k in ['auroc','accuracy','balanced_accuracy','detector_plane_mse']}})
    top=max(r['auroc'] for r in rows)
    eligible=[r for r in rows if r['auroc']>=top-SEARCH['auroc_tolerance']]
    winner=max(eligible,key=lambda r:(r['accuracy'],r['balanced_accuracy'],r['auroc'],-r['detector_plane_mse'],-SEARCH['candidates'].index(r['activation'])))
    selection={'selected':winner['activation'],'screen_summary':rows,'eligible':[r['activation'] for r in eligible],
      'rule':SEARCH['selection'],'test_used':False,'source_hashes':source}
    selection_path=ROOT/'protocol/activation_selection.json'
    if selection_path.exists():assert read(selection_path)==selection
    else:save(selection_path,selection)
    print('ACTIVATION_SELECTED '+json.dumps(selection),flush=True)
    formal=formal_variants(selection['selected']);assert len(formal)==12
    for seed in EXP['seeds']:
        for v in formal:
            index+=1;train_one(v,seed,train,val,index,source)
    # All thresholds selected on validation with an identical rule for all four model families.
    status(state='validation_calibration',formal_runs=60,selected_activation=selection['selected'])
    checkpoints={};threshold_hashes={};orders={}
    for seed in EXP['seeds']:
        for v in formal:
            dest=run_dir(v,seed);done=read(dest/'completed.json')
            if seed in orders:assert orders[seed]==done['order_sha256']
            else:orders[seed]=done['order_sha256']
            validation_predictions(v,seed,val)
            checkpoints[(dest/'best.pt').relative_to(ROOT).as_posix()]=sha(dest/'best.pt')
            threshold_hashes[(dest/'thresholds.json').relative_to(ROOT).as_posix()]=sha(dest/'thresholds.json')
    assert source==snapshot()
    save(ROOT/'protocol/final_lock.json',{'created_at':now(),'source_hashes':source,'checkpoints':checkpoints,
      'thresholds':threshold_hashes,'activation_selection_sha256':sha(selection_path),'formal_variants':formal})
    del train,val;torch.cuda.empty_cache();test=getdata('test');status(state='testing',formal_runs=60)
    for seed in EXP['seeds']:
        for v in formal:
            dest=run_dir(v,seed);ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
            model=build(v['architecture'],CONFIGS[v['id']]).cuda();model.load_state_dict(ck['model'],strict=True)
            result,rows=evaluate(model,test,True);csvwrite(dest/'test_predictions.csv',rows);save(dest/'test_metrics.json',result)
            y=np.array([r['label_true'] for r in rows]);p=np.array([[r['score0'],r['score1']] for r in rows])
            thresholds=read(dest/'thresholds.json')['policies']
            calibrated={k:full_metrics(y,p,r['threshold'],result['detector_plane_mse']) for k,r in thresholds.items()}
            for k in ['auroc','accuracy','mse','balanced_accuracy','macro_f1','positive_recall','specificity']:
                assert abs(calibrated['fixed_0.5'][k]-result[k])<1e-12
            save(dest/'test_threshold_metrics.json',calibrated)
            print(json.dumps({'test_variant':v['id'],'seed':seed,'auroc':result['auroc'],'accuracy':result['accuracy'],
              'val_threshold_test_accuracy':calibrated['val_accuracy']['accuracy']}),flush=True)
            del model,ck;torch.cuda.empty_cache()
    from report import make_report
    make_report();status(state='completed',formal_runs=60,screening_runs=12,unique_training_runs=68,
                        selected_activation=selection['selected'],report='reports/report.md')
    print('EXPERIMENT COMPLETED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','run','resume']);args=parser.parse_args()
    try:
        if args.action=='preflight':
            from preflight import preflight
            preflight()
        else:run(resume=args.action=='resume')
    except Exception as exc:
        status(state='failed',error=repr(exc));traceback.print_exc();raise
