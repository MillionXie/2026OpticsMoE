"""Validation-only threshold rules fixed before any test predictions are read."""
import numpy as np
from metrics import metrics

def at_threshold(y,p,t):
    y=np.asarray(y);pred=np.asarray(p)[:,1]>t
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    f0=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'threshold':float(t),'accuracy':(tn+tp)/len(y),
      'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2,
      'macro_f1':(f0+f1)/2,'positive_recall':tp/(tp+fn),'specificity':tn/(tn+fp),
      'confusion_matrix':[[tn,fp],[fn,tp]]}

def choose_thresholds(y,p):
    scores=np.unique(p[:,1]);midpoints=(scores[:-1]+scores[1:])/2
    candidates=np.unique(np.concatenate(([0.,.5,1.],midpoints)))
    rows=[at_threshold(y,p,t) for t in candidates]
    result={'fixed_0.5':at_threshold(y,p,.5)}
    for name,first,second in [('val_accuracy','accuracy','balanced_accuracy'),('val_balanced','balanced_accuracy','accuracy')]:
        result[name]=max(rows,key=lambda r:(r[first],r[second],-abs(r['threshold']-.5),-r['threshold']))
    return {'selection_split':'val','candidate_count':len(rows),'policies':result}

def full_metrics(y,p,t,plane_mse):
    out=metrics(y,p);out.update(at_threshold(y,p,t));out['detector_plane_mse']=plane_mse
    # Recompute all threshold-sensitive optional fields and warnings too.
    tn,fp=out['confusion_matrix'][0];fn,tp=out['confusion_matrix'][1]
    out['positive_precision']=tp/(tp+fp) if tp+fp else 0.
    out['positive_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    out['warnings']=['No positive predictions.'] if tp+fp==0 else []
    return out
