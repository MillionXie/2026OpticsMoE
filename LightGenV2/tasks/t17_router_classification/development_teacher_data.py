"""Explicit TRAIN/original-TEST membership for development teacher/student training."""
import numpy as np


def development_training_membership(data):
    """Deduplicate against VAL and earlier training pixels without changing evaluation."""
    validation=set(data['val']['hashes'].tolist());seen=set();keep=[];removed=[];offset=0
    for split in ('train','test'):
        for i,digest in enumerate(data[split]['hashes']):
            reason='validation_pixel_duplicate' if digest in validation else 'duplicate_training_pixel' if digest in seen else None
            if reason:removed.append(dict(split=split,sample_id=str(data[split]['ids'][i]),reason=reason))
            else:keep.append(offset+i);seen.add(digest)
        offset+=len(data[split]['hashes'])
    return np.asarray(keep,dtype=np.int64),removed
