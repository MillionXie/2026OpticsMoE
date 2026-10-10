"""Validation-only warm start of rho=.3; never changes historical training sources."""
import argparse
import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import run as t


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['train','evaluate'],required=True)
    p.add_argument('--depth',type=int,choices=[2,4,6],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--continuation-parent',action='store_true')
    p.add_argument('--budget',type=int,default=50)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
    a.out.mkdir(parents=True,exist_ok=False)
    t.torch.set_num_threads(4)
    ck=t.torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['depth']==a.depth and ck['config']['residual_rho']==.3
    src=dict(parent=t.source_identity(),continuation=t.r.sha(Path(__file__)))
    if a.phase=='train':
        assert 1<=a.budget<=50
        if a.continuation_parent:
            assert ck['sources']['parent']==t.source_identity()
            assert ck['sources']['continuation']=='3983cc1dd985194b6235f53ff5e017164b3134226214d7f55b898ab378e26f0f'
        else:
            assert ck['sources']==t.source_identity(), 'Parent source identity mismatch'
        cfg=copy.deepcopy(ck['config'])
        offset=ck['config'].get('continuation_augmentation_epoch_offset',0)+ck['epoch'] if a.continuation_parent else 100
        cfg.update(epochs=a.budget,minimum_epochs=min(15,a.budget),patience=12,lr=.0006,
                   training_profile='rho03_best_warmstart_lr0006_50',
                   parent_checkpoint_sha256=t.r.sha(a.checkpoint),
                   continuation_augmentation_epoch_offset=offset)
    else:
        # These published revisions share the unchanged forward/loss implementation;
        # the later revision adds only budget and warm-start lineage handling.
        archived={hashlib.sha256(subprocess.check_output(['git','show',rev+':LightGenV2/tasks/t18_optical_residual_ablation/continue_residual.py'])).hexdigest()
                  for rev in ['e6b125e9b','65917810b']}
        assert ck['sources']['parent']==t.source_identity()
        assert ck['sources']['continuation'] in archived | {src['continuation']}
        cfg=ck['config']
        assert cfg['training_profile']=='rho03_best_warmstart_lr0006_50'
    t.r.save(a.out/'metadata.json',dict(command=sys.argv,config=cfg,sources=src,
        checkpoint_training_sources=ck['sources'],
        parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],
        data_sha256=t.r.sha(a.data),git_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),time=t.r.now(),
        test_read=a.phase=='evaluate',scope='residual-only extra-budget exploratory',
        optimizer_restart=True,ema_restart_from_selected_parent=True))
    if a.phase=='evaluate':
        model=t.build('moe',a.depth,cfg);model.load_state_dict(ck['model'])
        vm,rows=t.b.evaluate(model,t.k.load_data(a.data,'val'),'moe',cfg['batch_size'])
        t.r.csvwrite(a.out/'val_predictions.csv',rows)
        tm,rows=t.b.evaluate(model,t.k.load_data(a.data,'test'),'moe',cfg['batch_size'])
        t.r.csvwrite(a.out/'test_predictions.csv',rows)
        t.r.save(a.out/'metrics.json',dict(val=vm,test=tm,epoch=ck['epoch'],
            checkpoint_sha256=t.r.sha(a.checkpoint),scope='exploratory_test_once'))
    else:
        original=t.b.build
        def warm_build(arch,depth,config):
            model=original(arch,depth,config)
            model.load_state_dict(ck['model'])
            return model
        t.b.build=warm_build
        order=t.r.epoch_order; affine=t.b.affine_parameters
        t.r.epoch_order=lambda seed,epoch,n:order(seed,epoch+offset,n)
        t.b.affine_parameters=lambda n,seed,epoch,aug:affine(n,seed,epoch+offset,aug)
        data=t.k.load_data(a.data,'train');val=t.k.load_data(a.data,'val')
        result=t.b.train('moe',a.depth,17,data,val,cfg,a.out,src)
        initial=t.r.read(a.out/result['name']/'initial_validation.json')
        improved=result['metrics']['val']['balanced_nll']<initial['balanced_nll']-cfg['min_delta']
        result.update(parent_selected_epoch=ck['epoch'],parent_validation=initial,
                      improves_parent=improved,selection='validation balanced NLL, parent remains eligible')
        t.r.save(a.out/'result.json',result)
    t.r.save(a.out/'status.json',dict(state='complete',phase=a.phase,time=t.r.now()))


if __name__=='__main__':
    main()
