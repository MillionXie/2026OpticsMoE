"""Re-evaluate saved A/B checkpoints on the sealed validation images only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import OpticalMoE
from .data import load,sha
from .run import evaluate,save

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    torch.set_num_threads(4)
    cfg=json.loads((a.run/'config.json').read_text()); meta=json.loads((a.run/'metadata.json').read_text()); expected=json.loads((a.run/'metrics.json').read_text())
    if meta['data_sha256']!=sha(a.data): raise ValueError('Different dataset')
    data,_=load(a.data,a.manifest,cfg['seed']); x=torch.from_numpy(data['val_images']); y=torch.from_numpy(data['val_labels']).long()
    model=OpticalMoE(cfg).to(a.device); results={}; hashes={}
    for stage in ('A','B'):
        path=a.run/stage/'best_checkpoint.pt'; hashes[stage]=sha(path)
        checkpoint=torch.load(path,map_location=a.device,weights_only=False); model.load_state_dict(checkpoint['model'])
        cases=[('A_before','A',None)] if stage=='A' else [(task+'_'+label,task,mask) for task in ('A','B') for label,mask in [('all',None),('old_only',[True]*4+[False]*8),('new_only',[False]*4+[True]*4+[False]*4)]]
        for key,task,mask in cases:
            metrics,_,_=evaluate(model,x,y,task,cfg['batch_size'],mask)
            if abs(metrics['accuracy']-expected[key]['accuracy'])>1e-7 or metrics['confusion']!=expected[key]['confusion']: raise RuntimeError('Checkpoint re-evaluation mismatch: '+key)
            results[key]=metrics
    save(a.run/'reevaluation.json',dict(matched=True,checkpoint_sha256=hashes,data_sha256=meta['data_sha256'],results=results))
    print(json.dumps(dict(matched=True,checkpoint_sha256=hashes)))
if __name__=='__main__': main()
