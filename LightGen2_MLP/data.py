from pathlib import Path
import csv

import numpy as np
import torch


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
