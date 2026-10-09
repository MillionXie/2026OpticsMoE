"""Evaluate a validation-selected exploratory candidate once, with a receipt guard."""
import argparse
import json
import numpy as np
import torch
import shutil
from pathlib import Path
from .train import encode,evaluate,write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain,sha256

def reuse_parent_receipt(run,ck):
    if ck.get('weights_kind')!='parent':return False
    cfg=ck['config'];parent=Path(cfg['init_run'])
    source=parent/'best_checkpoint.pt'
    if sha256(source)!=cfg['initial_checkpoint']['sha256']:raise ValueError('parent changed')
    previous=torch.load(source,map_location='cpu',weights_only=False)
    if set(previous['model'])!=set(ck['model']) or any(
        not torch.equal(previous['model'][name].cpu(),value.cpu()) for name,value in ck['model'].items()):
        raise ValueError('parent weights are not identical')
    result=json.loads((parent/'result.json').read_text())
    if result['checkpoint_sha256']!=sha256(source):raise ValueError('parent test receipt mismatch')
    result.update(checkpoint_sha256=sha256(run/'best_checkpoint.pt'),test_evaluations=0,
        source_test_checkpoint_sha256=sha256(source),test_receipt_reused_from=str(parent),
        interpretation='identical parent weights; no new test inference')
    shutil.copyfile(parent/'test_predictions.npz',run/'test_predictions.npz')
    write(run/'result.json',result)
    write(run/'status.json',dict(state='completed_reused_test',result=result))
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    target=a.run/'result.json'
    if target.exists():raise FileExistsError('already evaluated')
    ck=torch.load(a.run/'best_checkpoint.pt',map_location='cuda',weights_only=False)
    cfg=ck['config']
    if not cfg['skip_test']:raise ValueError('only validation-only candidates')
    if cfg.get('router_features')=='centered':raise ValueError('stopped unauthorized historical candidate')
    if reuse_parent_receipt(a.run,ck):
        print('Identical parent weights: reused existing test receipt without inference');return
    if sha256(Path(cfg['data']))!=cfg['data_sha256'] or sha256(Path(cfg['manifest']))!=cfg['manifest_sha256']:
        raise ValueError('dataset or manifest changed')
    data,_=load_domain(cfg['data'],cfg['manifest'])
    profile=cfg.get('profile','sixteen_dense')
    if profile in ('four_top2','four_top2_ccd'):
        from .model_four import FourRouterClassification
        model=FourRouterClassification(cfg['architecture'],
            'ccd_grid' if profile=='four_top2_ccd' else 'linear').cuda()
    else:
        from .model import RouterClassification
        model=RouterClassification(cfg['architecture'],cfg['router_features']).cuda()
    model.load_state_dict(ck['model'])
    result,pred,q=evaluate(model,encode(data['test_images'],profile).cuda(),
        torch.as_tensor(data['test_labels'],device='cuda',dtype=torch.long),cfg['batch'])
    result.update(selected_epoch=ck['epoch'],checkpoint_sha256=sha256(a.run/'best_checkpoint.pt'),
                  test_evaluations=1,interpretation='post-hoc exploratory; baseline test previously seen')
    if 'fine_tune_epoch' in ck:
        result.update(fine_tune_epoch=ck['fine_tune_epoch'],weights_kind=ck['weights_kind'])
    np.savez_compressed(a.run/'test_predictions.npz',ids=data['test_ids'],labels=data['test_labels'],
                        predictions=pred,route_power=q)
    write(target,result);write(a.run/'status.json',{'state':'completed_exploratory','result':result})
    print(json.dumps(result))

if __name__=='__main__':main()
