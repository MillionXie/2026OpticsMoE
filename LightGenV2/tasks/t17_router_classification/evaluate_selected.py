"""Evaluate a validation-selected exploratory candidate once, with a receipt guard."""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from .model import RouterClassification
from .train import encode,evaluate,write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    target=a.run/'result.json'
    if target.exists():raise FileExistsError('already evaluated')
    ck=torch.load(a.run/'best_checkpoint.pt',map_location='cuda',weights_only=False)
    cfg=ck['config']
    if not cfg['skip_test']:raise ValueError('only validation-only candidates')
    data,_=load_domain(cfg['data'],cfg['manifest'])
    model=RouterClassification(cfg['architecture'],cfg['router_features']).cuda()
    model.load_state_dict(ck['model'])
    result,pred,q=evaluate(model,encode(data['test_images']).cuda(),
        torch.as_tensor(data['test_labels'],device='cuda',dtype=torch.long),cfg['batch'])
    result.update(selected_epoch=ck['epoch'],checkpoint_sha256=sha256(a.run/'best_checkpoint.pt'),
                  test_evaluations=1,interpretation='post-hoc exploratory; baseline test previously seen')
    np.savez_compressed(a.run/'test_predictions.npz',ids=data['test_ids'],labels=data['test_labels'],
                        predictions=pred,route_power=q)
    write(target,result);write(a.run/'status.json',{'state':'completed_exploratory','result':result})
    print(json.dumps(result))

if __name__=='__main__':main()
