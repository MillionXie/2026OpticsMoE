from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch
from scipy.stats import spearmanr

if __package__:
    from .model import load_lightgen2
    from .data import read_input_metadata,read_reference_scores,iter_rgb_batches
    from .plots import configure_figures,export_results,export_input_frames
else:
    from model import load_lightgen2
    from data import read_input_metadata,read_reference_scores,iter_rgb_batches
    from plots import configure_figures,export_results,export_input_frames


PROJECT=Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate(model,archive,names,targets,device,directory,preview_batch=1):
    batches=math.ceil(len(names)/16)
    if not 1<=preview_batch<=batches:raise ValueError('Preview batch is outside the input list')
    records=[]
    preview=None
    for number,(start,count,images) in enumerate(iter_rgb_batches(archive,names),1):
        images=images.to(device)
        output=model(normalized_rgb=images)
        values=output['prediction'][0,:count].cpu().tolist()
        records.extend(dict(sample_id=names[start+index],target_MOS=float(targets[start+index]),predicted_MOS=value)
                       for index,value in enumerate(values))
        if number==preview_batch:preview=(images.cpu(),count)
        print(f'Batch {number}/{batches}  videos {len(records)}/{len(names)}',flush=True)
    prediction=np.asarray([row['predicted_MOS'] for row in records])
    score=float(spearmanr(prediction,targets).statistic) if np.ptp(prediction)>0 and np.ptp(targets)>0 else None
    metrics=dict(videos=len(records),SRCC=score)
    export_results(records,metrics,directory)
    export_input_frames(*preview,directory)
    return records,metrics


def main():
    parser=argparse.ArgumentParser(description='Run LightGen-II MLP video quality inference.')
    parser.add_argument('--weights',type=Path,default=PROJECT/'weights/LightGen2.pt')
    parser.add_argument('--input',type=Path,default=PROJECT/'data/input_for_review.npz')
    parser.add_argument('--reference-scores',type=Path,default=PROJECT/'data/reference_mos.csv')
    parser.add_argument('--output',type=Path,default=PROJECT/'results')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--preview-batch',type=int,default=1)
    args=parser.parse_args()
    torch.set_num_threads(4)
    configure_figures()
    model,_=load_lightgen2(args.weights,args.device)
    with np.load(args.input,allow_pickle=False) as archive:
        names=read_input_metadata(archive)
        targets=read_reference_scores(args.reference_scores,names)
        _,metrics=evaluate(model,archive,names,targets,args.device,args.output,args.preview_batch)
    target=args.output/'02_metrics.json'
    temporary=target.with_suffix('.json.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(metrics,stream,indent=2,allow_nan=False)
        stream.write('\n')
    temporary.replace(target)
    print(f"SRCC {metrics['SRCC']:.2f} over {metrics['videos']} videos",flush=True)
    print(f'Saved results to {args.output}',flush=True)


if __name__=='__main__':main()
