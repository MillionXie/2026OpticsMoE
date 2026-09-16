"""Retrospective, all-model checkpoint-policy diagnostic; no optimization steps.

Evaluate every epoch-50 checkpoint, rather than selecting favorable models from
this alternative. The original validation-selected results remain authoritative
for their declared protocol and are never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
TASK = Path(__file__).resolve().parents[1]
ARCHIVE = TASK/'adrenal_softsign_code_export_20260915_145336/code'
sys.path.insert(0,str(ARCHIVE))
import numpy as np
import torch
from torch.nn import functional as F
import run_experiment as r
from calibration import choose_thresholds, full_metrics


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path)
    group.add_argument('--original-depth-audit',type=Path)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    r.EXP['data_npz']=str(a.data.resolve())
    r.setup()
    a.out.mkdir(parents=True,exist_ok=False)
    if a.run:
        parent_metadata=a.run/'metadata.json'
        parent=r.read(parent_metadata)
        summaries=r.read(a.run/'validation_results.json')
        assert parent['seeds']==[17] and parent['protocol']['epochs']==50
        for item in summaries:
            source=a.run/'runs'/item['variant']/'seed17'
            item.update(source_directory=str(source),checkpoint_filename='last_checkpoint.pt',
                        model_config=r.read(source/'config.json'))
    else:
        parent_metadata=a.original_depth_audit/'metadata.json'
        summaries=[]
        for item in r.read(a.original_depth_audit/'results.json'):
            source=Path(item['source_run'])
            summaries.append(dict(variant=item['variant'],
                source_directory=str(source/'runs'/item['variant']/'seed17'),checkpoint_filename='last.pt',
                model_config=r.read(source/'configs.json')[item['variant']],
                last_train=item['metrics']['last_train'],last_checkpoint_sha256=item['checkpoint_sha256']['last']))
    assert len(summaries)==12
    lock={}
    for item in summaries:
        path=Path(item['source_directory'])/item['checkpoint_filename']
        assert r.sha(path)==item['last_checkpoint_sha256']
        lock[str(path.resolve())]=r.sha(path)
    r.save(a.out/'checkpoint_lock.json',dict(checkpoints=lock,epoch=50,time=r.now(),
           scope='all twelve models; no per-model choice between best and last',
           test_status='retrospective diagnostic proposed after validation-selected test scores were inspected'))
    r.save(a.out/'configs.json',{item['variant']:item['model_config'] for item in summaries})
    r.save(a.out/'metadata.json',dict(command=sys.argv,parent_metadata=str(parent_metadata.resolve()),
           parent_metadata_sha256=r.sha(parent_metadata),data_sha256=r.sha(a.data),
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           source_sha256=r.sha(__file__),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
           environment_reference=str(parent_metadata.resolve())))
    val=r.getdata('val')
    # The complete alternative checkpoint set is locked before this access.
    with np.load(a.data,allow_pickle=False) as z:
        x,y,ids=z['test_images'].copy(),z['test_labels'].reshape(-1).copy(),z['test_ids'].copy()
    assert np.bincount(y).tolist()==[229,69] and len(set(ids))==298
    x=F.interpolate(torch.from_numpy(x[:,None]),size=(100,100),mode='bicubic',
                    align_corners=False,antialias=True).clamp(0,1).cuda()
    test=x,torch.from_numpy(y).long().cuda(),ids
    results=[]
    for item in summaries:
        source=Path(item['source_directory'])
        checkpoint_path=source/item['checkpoint_filename']
        dest=a.out/item['variant'];dest.mkdir()
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert ck['epoch']==50 and ck['seed']==17
        model=r.build(ck['variant']['architecture'],item['model_config']).cuda()
        model.load_state_dict(ck['model'])
        before={n:r.sha_tensor(p) for n,p in model.named_parameters()}
        vm,vr=r.evaluate(model,val,predictions=True)
        r.csvwrite(dest/'validation_predictions.csv',vr)
        vy=np.array([row['label_true'] for row in vr])
        vp=np.array([[row['score0'],row['score1']] for row in vr])
        thresholds=choose_thresholds(vy,vp)
        r.save(dest/'thresholds.json',thresholds)
        tm,tr=r.evaluate(model,test,predictions=True)
        r.csvwrite(dest/'test_predictions.csv',tr)
        threshold=thresholds['policies']['val_balanced']['threshold']
        tp=np.array([[row['score0'],row['score1']] for row in tr])
        calibrated=full_metrics(y,tp,threshold,tm['detector_plane_mse'])
        assert before=={n:r.sha_tensor(p) for n,p in model.named_parameters()}
        result=dict(variant=item['variant'],seed=17,epoch=50,checkpoint_sha256=lock[str(checkpoint_path.resolve())],
                    train=item['last_train'],val=vm,test=tm,val_threshold_test=calibrated,
                    weights_unchanged=True)
        r.save(dest/'metrics.json',result)
        results.append(result)
        del model,ck
        torch.cuda.empty_cache()
    r.save(a.out/'results.json',results)
    r.save(a.out/'status.json',dict(state='complete',models=12,optimizer_steps=0))
    print(json.dumps([dict(variant=x['variant'],val=x['val']['auroc'],test=x['test']['auroc']) for x in results]))


if __name__=='__main__':
    main()
