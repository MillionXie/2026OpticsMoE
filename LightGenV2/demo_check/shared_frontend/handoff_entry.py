"""Standalone release entry template; dependencies are exported by build_lab_package.py."""
import argparse
import csv
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import time
import numpy as np
import torch
import utils as base
from utils import save,sha,tensors_sha,batches
from optical_model import objective
from models import FrontendOptics,SharedFrontend
from training import train

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['evaluate','smoke','train'],required=True)
    p.add_argument('--data',type=Path,default=HERE/'data/data.npz')
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--weights-root',type=Path,default=HERE/'checkpoints')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='train from each architecture last checkpoint, preserving Adam moments')
    p.add_argument('--epochs',type=int,help='total optical epochs, including already completed epochs')
    p.add_argument('--learning-rate',type=float)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text());ocfg=json.loads((HERE/'optical_config.json').read_text())
    if args.epochs is not None:cfg['epochs']=args.epochs
    if args.learning_rate is not None:cfg['learning_rate']=args.learning_rate
    assert cfg['architectures']==['dynamic_four','full_d2nn']
    if args.resume:assert args.mode=='train' and cfg['epochs']>20
    ocfg['seed']=cfg['seed'];ocfg['architectures']=cfg['architectures']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.manual_seed(cfg['seed'])
    assert torch.cuda.is_available(),'CUDA required for this matched reproduction entry'
    provenance=json.loads((HERE/'PROVENANCE.json').read_text())
    assert sha(args.data)==provenance['data_sha256'],'Use the included matched subset for reproduction'
    with np.load(args.data,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    assert not any(k.startswith('test') for k in arrays)
    train_data=tuple(torch.from_numpy(arrays['train_'+k]) for k in ['images','labels','domains'])
    val=tuple(torch.from_numpy(arrays['validation_'+k]) for k in ['images','labels','domains'])
    frontend_file=args.weights_root/'frontend/best_checkpoint.pt'
    assert sha(frontend_file)==provenance['frontend_checkpoint_sha256']
    frontend=SharedFrontend(torch.load(frontend_file,map_location='cpu',weights_only=False)['model']).cuda()
    frozen=tensors_sha(frontend.state_dict())
    save(args.out/'metadata.json',dict(config=cfg,optical_config=ocfg,command=sys.argv,python=sys.version,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),provenance=provenance,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        data_sha256=sha(args.data),frontend_sha256=sha(frontend_file),resume=args.resume,test_set_used=False))
    if args.mode=='train':
        train(frontend,cfg,ocfg,arrays,train_data,val,args.out,resume=args.weights_root if args.resume else None)
    else:
        reports=[]
        for architecture in cfg['architectures']:
            model=FrontendOptics(frontend,architecture,ocfg).cuda()
            checkpoint_path=args.weights_root/architecture/'best_checkpoint.pt'
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            assert sha(checkpoint_path)==provenance['best_checkpoint_sha256'][architecture]
            model.optical.load_state_dict(checkpoint['model'])
            if args.mode=='evaluate':
                metrics,prob=base.evaluate(model,val,cfg)
                expected=json.loads((HERE/'reference'/architecture/'summary.json').read_text())
                assert metrics==expected['validation']
                with (HERE/'reference'/architecture/'validation_predictions.csv').open() as f:rows=list(csv.DictReader(f))
                assert [r['sample_id'] for r in rows]==arrays['validation_ids'].tolist()
                original=np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows],dtype=np.float32)
                assert np.array_equal(prob,original),'Prediction differs; record environment and numerical difference before claiming exact reproduction'
                reports.append(dict(architecture=architecture,selected_epoch=checkpoint['epoch'],validation=metrics,predictions_bitwise_identical=True))
            else:
                model.train();images=train_data[0][:8].cuda();labels=train_data[1][:8].cuda();output=model(images)
                loss=objective(output,labels);loss.backward()
                gradients={n:float(p.grad.norm()) for n,p in model.optical.named_parameters()}
                assert all(np.isfinite(v) and v>0 for v in gradients.values())
                assert all(p.grad is None for p in frontend.parameters()) and not frontend.training
                last=torch.load(args.weights_root/architecture/'last_checkpoint.pt',map_location='cpu',weights_only=False)
                assert last['epoch']==20
                optimizer=torch.optim.Adam(model.optical.parameters(),lr=.001);optimizer.load_state_dict(last['optimizer'])
                assert {int(s['step']) for s in optimizer.state.values()}=={3760}
                reports.append(dict(architecture=architecture,gradients=gradients,frozen_frontend=True,resume_epoch=20,adam_step=3760))
            assert frozen==tensors_sha(frontend.state_dict())
            del model;torch.cuda.empty_cache()
        save(args.out/'results.json',reports);print(json.dumps(reports),flush=True)
    save(args.out/'status.json',dict(state='complete',mode=args.mode))


if __name__=='__main__':main()
