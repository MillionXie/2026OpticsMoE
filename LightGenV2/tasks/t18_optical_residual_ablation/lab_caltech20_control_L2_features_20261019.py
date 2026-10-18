"""Verify the completed L2 nores capture and export exact original-head inputs.

No devices, new data, new architecture or adaptation here. The existing 275/411
parent is never modified. Complete measured-graph replay on CPU is mandatory.
"""
import argparse,json,shutil,sys,traceback,types
from pathlib import Path
import numpy as np
import torch
from lab_caltech20_rich_readout_20261018 import load_features,existing_head,evaluate
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images
from lab_caltech10_20261015 import save,sha


def run(a):
    torch.set_num_threads(2)
    capture=a.session/'resume_chain_20261018/full_control'
    release,spec,original,_,data,files=load_features(types.SimpleNamespace(
        release=a.session/'release.json',capture=capture,assets=a.session/'assets',arm='control'))
    source=a.selected or a.session/'control_head_gap6_20261018'
    report=json.loads((source/'report.json').read_text());assert report['state']=='complete'
    warm=source/'best.pt';digest=sha(warm);assert digest==report['results']['best']['PT_SHA256']
    if a.selected is None:assert digest=='2fd7542bea04b45ce60c4370a1fb2eb52d275c331cf40d8b799e7ea15da033f0'
    state=torch.load(warm,map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
    model,_=load_checkpoint(a.session/'assets'/spec['file'],spec['sha256'],2,0.,alpha_schema=spec['alpha_schema'])
    model.load_state_dict(state,strict=True);head=existing_head(state);proof=[];arrays={};results={}
    if a.selected is None:assert not a.out.exists();a.out.mkdir();(a.out/'selected_features').mkdir()
    dm=json.loads((a.session/'assets/data_manifest.json').read_text())
    with np.load(a.session/'assets'/dm['file'],allow_pickle=False) as identity,torch.inference_mode():
        for role,n in [('train',1000),('test',411)]:
            fields={}
            for stage in ('router','0','1'):
                path=capture/'control'/role/stage/'ccd.npy';arrays[role+'/'+stage]=sha(path)
                fields[stage]=np.load(path,mmap_mode='r');assert fields[stage].shape==(n,478,478)
            images=torch.from_numpy(identity[role+'_images'].transpose(0,3,1,2).copy());features=[];prob=[]
            for begin in range(0,n,8):
                def measured(stage,amplitude,phase,ids):return np.asarray(fields[stage][begin:begin+8]).copy()
                p=model(encode_images(images[begin:begin+8]),callback=measured,ids=identity[role+'_ids'][begin:begin+8])[0]
                features.append(model.readout['features'].numpy().copy());prob.append(p.numpy())
            x=np.concatenate(features);prob=np.concatenate(prob);feature_error=float(np.abs(x-data[role]['x'].numpy()).max())
            error=float(np.abs(prob-head(torch.from_numpy(x)).softmax(1).numpy()).max());assert feature_error<=3e-6 and error<=3e-6
            m,p=evaluate(head,data[role]);assert m['accuracy']==report['results']['best']['TRAIN' if role=='train' else 'TEST_development']['accuracy']
            assert int((prob.argmax(1)==identity[role+'_labels']).sum())==round(n*m['accuracy'])
            proof.append(dict(role=role,samples=n,correct=round(n*m['accuracy']),feature_error=feature_error,probability_error=error));results[role]=m
            if a.selected is None:
                folder=a.out/'selected_features'/role;folder.mkdir();np.savez_compressed(folder/'predictions.npz',features=x,labels=identity[role+'_labels'],ids=identity[role+'_ids'],probabilities=p)
                np.savez_compressed(a.out/('best_'+role.upper()+'_predictions.npz'),labels=identity[role+'_labels'],ids=identity[role+'_ids'],probabilities=p)
    assert sha(warm)==digest and all(sha(capture/'control'/role/'predictions.npz')==d for role,d in files.items())
    audit=dict(passed=True,fresh_complete_measured_graph=True,depth=2,proof=proof,CCD_arrays_SHA256=arrays,selected_PT_SHA256=digest,all_nonhead_parameters_unchanged=True,SDK=False,GPU=False,new_CCD=0,TRAIN=1000,TEST=411,VALID=0)
    if a.selected is not None:save(source/'audit_fullgraph.json',audit)
    else:
        shutil.copy2(warm,a.out/'best.pt');assert sha(a.out/'best.pt')==digest
        ff={r:sha(a.out/'selected_features'/r/'predictions.npz') for r in ('train','test')}
        save(a.out/'selected_features/report.json',dict(passed=True,features_SHA256=ff,proof=proof,original_features_SHA256=files))
        with np.load(a.out/'best_TEST_predictions.npz',allow_pickle=False) as z:nll=float(-np.log(z['probabilities'][np.arange(411),z['labels']].clip(1e-12)).mean())
        save(a.out/'report.json',dict(state='complete',depth=2,best_correct=round(411*results['test']['accuracy']),best_NLL=nll,source_run=str(source),results=dict(best=dict(TRAIN=results['train'],TEST_development=results['test'],PT_SHA256=digest)),no_training=True,SDK=False,GPU=False))
        save(a.out/'audit_fullgraph.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--selected',type=Path);a=p.parse_args()
    if sys.stdout is None:sys.stdout=(a.selected or a.out).with_suffix('.log').open('x',encoding='utf8',buffering=1);sys.stderr=sys.stdout
    try:run(a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
