"""Bounded, single-GPU training-method comparison on the SAME 446/112 split."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time

from .adapt_measured_readout import train, split_indices
from .lab_runtime import read, write, sha


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','checkpoint','reference-split','reference-result','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/spatial_hardware_readout_tuning.json'))
    p.add_argument('--device',default='cuda');a=p.parse_args()
    c=read(a.config);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    if c['train_fraction']!=.8:raise ValueError('This comparison must keep the 20% holdout')
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    train_idx,hold_idx=split_indices(len(data['video_ids']),.8,c['seed']);reference=read(a.reference_split)
    if reference['train']!=[data['video_ids'][i] for i in train_idx] or reference['holdout']!=[data['video_ids'][i] for i in hold_idx]:
        raise ValueError('Split changed relative to previous 80% experiment')
    del data
    out.mkdir(parents=True);write(out/'config.json',c)
    previous=read(a.reference_result)
    summary={'previous':{k:v for k,v in previous.items() if k!='rows'}}
    try:
        for trial in c['trials']:
            name=trial['name']
            if not name.replace('_','').isalnum():raise ValueError('Invalid trial ID')
            write(out/'status.json',dict(state='training',trial=name,completed=list(summary),updated=time.strftime('%Y-%m-%dT%H:%M:%S')))
            kwargs={k:v for k,v in trial.items() if k!='name'}
            train(SimpleNamespace(cache=a.cache,checkpoint=a.checkpoint,output=str(out/name),device=a.device,
                                  epochs=c['epochs'],seed=c['seed'],train_fraction=.8,**kwargs))
            summary[name]=read(out/name/'results.json')['split80'];write(out/'comparison.json',summary)
        winner=max(summary,key=lambda k:(summary[k]['after']['selection']['srcc'],-summary[k]['after']['selection']['rmse']))
        checkpoint=Path(a.reference_result).parent/'best_checkpoint.pt' if winner=='previous' else out/winner/'split80/best_checkpoint.pt'
        write(out/'selected.json',dict(candidate=winner,checkpoint=str(checkpoint),sha256=sha(checkpoint),
              metrics=summary[winner]['after'],independent_test=False,holdout_used_for_epoch_and_trial_selection=True))
        write(out/'status.json',dict(state='complete',winner=winner,results=summary))
    except Exception as e:
        write(out/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':main()
