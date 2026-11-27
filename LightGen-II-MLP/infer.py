from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch
from scipy.stats import spearmanr

if __package__:
    from .model import load_lightgen2
    from .visualize import configure_figures,export_results,export_input_frames
else:
    from model import load_lightgen2
    from visualize import configure_figures,export_results,export_input_frames


PROJECT=Path(__file__).resolve().parent


def read_input_metadata(archive):
    if int(archive['format_version']) not in (1,2):
        raise ValueError('Unsupported video-frame archive version')
    names=archive['sample_ids'].tolist()
    if not isinstance(names,list) or len(names)<2:
        raise ValueError('The input archive requires at least two videos')
    if any(not isinstance(name,str) or not name for name in names):
        raise ValueError('Video identities must be nonempty strings')
    if len(set(names))!=len(names):
        raise ValueError('Video identities must be unique')
    return names


def read_reference_scores(path,names):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['sample_id','target_MOS']:
            raise ValueError('Reference scores require sample_id and target_MOS columns')
        rows=list(reader)
    scores={row['sample_id']:float(row['target_MOS']) for row in rows}
    if len(scores)!=len(rows) or set(scores)!=set(names):
        raise ValueError('Reference identities must match the input videos exactly')
    result=np.asarray([scores[name] for name in names],dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError('Reference MOS values must be finite')
    return result


def iter_rgb_batches(archive,names):
    first_frames=None
    for number,start in enumerate(range(0,len(names),16),1):
        count=min(16,len(names)-start)
        frames=archive[f'rgb_{number:04d}']
        if frames.shape!=(count,4,3,448,448) or frames.dtype!=np.float16:
            raise ValueError(f'Batch {number} requires float16 frames shaped [{count},4,3,448,448]')
        if not np.isfinite(frames).all() or frames.min()<-1 or frames.max()>1:
            raise ValueError('RGB input values must be finite and lie in [-1,1]')
        if first_frames is None:first_frames=frames[:15].copy()
        if count<16:
            frames=np.concatenate((frames,first_frames[np.arange(16-count)%len(first_frames)]))
        yield start,count,torch.from_numpy(frames)[None]


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
    parser=argparse.ArgumentParser(description='Run LightGen-II video quality inference.')
    parser.add_argument('--weights',type=Path,default=PROJECT/'weights/LightGen2.pt')
    parser.add_argument('--input',type=Path,default=PROJECT/'data/input_for_review.npz')
    parser.add_argument('--reference-scores',type=Path,default=PROJECT/'data/reference_mos.csv')
    parser.add_argument('--output',type=Path,default=PROJECT/'results')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--visualize-field','--preview-batch',dest='preview_batch',type=int,default=1,
                        help='One-based batch number used for the input-frame preview')
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
