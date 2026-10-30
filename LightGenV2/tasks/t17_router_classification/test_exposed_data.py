"""Data access for explicitly TEST-exposed development runs only."""
import hashlib
import numpy as np
from PIL import Image
from .train_optical_four_ccd import prepare_dataset
from .train_linear_ccd_8class import prepare_index


def load_raw_splits(root,cfg,splits=('train','val','test')):
    result={}
    if cfg['dataset']=='crc9':
        _,ids,identity,_=prepare_dataset(root,cfg)
        with np.load(root/'A_original.npz',allow_pickle=False) as z:
            for split in splits:
                images=z[split+'_images'];labels=z[split+'_labels']
                result[split]=dict(images=images,labels=labels,ids=ids[split],
                                   hashes=np.array([hashlib.sha256(im.tobytes()).hexdigest() for im in images]))
    else:
        rows,ids,identity=prepare_index(root,cfg)
        for split in splits:
            images=[]
            for row in rows[split]:
                with Image.open(root/row['relative_path']) as im:array=np.array(im.convert('RGB'))
                if hashlib.sha256(array.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('pixel SHA changed')
                images.append(array)
            result[split]=dict(images=np.stack(images),labels=np.array([int(r['class_id']) for r in rows[split]]),
                               ids=ids[split],hashes=np.array([r['pixel_sha256'] for r in rows[split]]))
    return result,identity


def metrics_from_scores(labels,scores):
    classes=scores.shape[1];pred=scores.argmax(1)
    cm=np.bincount(labels*classes+pred,minlength=classes*classes).reshape(classes,classes)
    recalls=cm.diagonal()/np.maximum(cm.sum(1),1)
    return dict(accuracy=float((pred==labels).mean()),balanced_accuracy=float(recalls.mean()),
                cross_entropy=float(-np.log(np.maximum(scores[np.arange(len(labels)),labels],1e-12)).mean()),
                confusion_matrix=cm.tolist(),per_class_recall=recalls.tolist(),n=len(labels))
