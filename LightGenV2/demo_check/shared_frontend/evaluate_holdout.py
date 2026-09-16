"""Lock validation-selected original/continued checkpoints before test evaluation."""
import argparse
import csv
import importlib.util
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('holdout_runner',HERE/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--continued',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    selected=[]
    for label,run in [('original',a.original),('continued',a.continued)]:
        assert json.loads((run/'status.json').read_text())['state']=='complete'
        for architecture in ['dynamic_four','full_d2nn']:
            summary=json.loads((run/architecture/'summary.json').read_text());path=run/architecture/'best_checkpoint.pt'
            assert r.sha(path)==summary['checkpoint_sha256']
            selected.append(dict(label=label,architecture=architecture,path=str(path),checkpoint_sha256=r.sha(path),
                                 selected_epoch=summary['selected_epoch'],validation=summary['validation'],train_unaugmented=summary['train_unaugmented']))
    assert r.sha(a.original/'frontend/best_checkpoint.pt')==r.sha(a.continued/'frontend/best_checkpoint.pt')
    # This immutable run's lock is written before reading any test arrays or metrics.
    r.save(a.out/'selection_lock.json',dict(selection='validation only; no further tuning in this experiment',models=selected,
                                         frontend_sha256=r.sha(a.original/'frontend/best_checkpoint.pt')))
    source=json.loads((a.original/'metadata.json').read_text());cfg=source['config'];ocfg=source['optical_config']
    manifest=json.loads(a.data.with_name('manifest.json').read_text());assert r.sha(a.data)==manifest['data_sha256']
    assert manifest['original_split_sha256']==source['split_sha256'] and manifest['train_validation_spatial_overlap']==0
    with np.load(a.data,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    assert all(k.startswith('test_') for k in arrays)
    test=tuple(torch.from_numpy(arrays['test_'+k]) for k in ['images','labels','domains'])
    for domain in (0,1):assert torch.bincount(test[1][test[2]==domain],minlength=10).tolist()==[100]*10
    frontend=r.SharedFrontend(torch.load(a.original/'frontend/best_checkpoint.pt',map_location='cpu',weights_only=False)['model']).cuda()
    frozen=r.tensors_sha(frontend.state_dict())
    r.save(a.out/'metadata.json',dict(command=sys.argv,config=cfg,optical_config=ocfg,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),data_sha256=r.sha(a.data),data_manifest_sha256=r.sha(a.data.with_name('manifest.json')),
        operation='single locked original-test-subset evaluation; no optimizer or test selection'))
    shutil.copyfile(a.data.with_name('manifest.json'),a.out/'test_manifest.json');shutil.copyfile(a.data,a.out/'test_data.npz')
    results=[]
    for entry in selected:
        model=r.FrontendOptics(frontend,entry['architecture'],ocfg).cuda();ck=torch.load(entry['path'],map_location='cpu',weights_only=False)
        assert ck['frontend_tensors_sha256']==frozen;model.optical.load_state_dict(ck['model'])
        metrics,prob=r.base.evaluate(model,test,cfg)
        with (a.out/(entry['label']+'_'+entry['architecture']+'_predictions.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['sample_id','domain','label','prediction']+[f'p{i}' for i in range(10)])
            w.writerows([str(i),int(d),int(y),int(p.argmax()),*p.tolist()] for i,d,y,p in zip(arrays['test_ids'],arrays['test_domains'],arrays['test_labels'],prob))
        assert frozen==r.tensors_sha(frontend.state_dict())
        results.append(dict(**entry,test=metrics));print(json.dumps(dict(label=entry['label'],architecture=entry['architecture'],test_accuracy=metrics['accuracy'])),flush=True)
        del model;torch.cuda.empty_cache()
    r.save(a.out/'results.json',results);r.save(a.out/'status.json',dict(state='complete'))


if __name__=='__main__':main()
