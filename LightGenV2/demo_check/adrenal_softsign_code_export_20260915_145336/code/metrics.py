import csv
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,balanced_accuracy_score,f1_score,precision_score,recall_score

def metrics(y,scores):
    y=np.asarray(y,dtype=int);p=np.asarray(scores,dtype=float)
    if p.shape!=(len(y),2) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
    if set(np.unique(y))!={0,1}:raise ValueError('AUROC undefined: both labels are required')
    pred=(p[:,1]>.5).astype(int);cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    warnings=[]
    if tp+fp==0:warnings.append('No positive predictions: positive precision defined as 0 (zero_division=0).')
    return {'n':len(y),'support':np.bincount(y,minlength=2).tolist(),'mse':float(np.mean((p-np.eye(2)[y])**2)),
        'accuracy':float(np.mean(y==pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'auroc':float(roc_auc_score(y,p[:,1])),'average_precision':float(average_precision_score(y,p[:,1])),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'positive_precision':float(precision_score(y,pred,zero_division=0)),
        'positive_recall':float(recall_score(y,pred,zero_division=0)),
        'positive_f1':float(f1_score(y,pred,zero_division=0)),
        'specificity':tn/(tn+fp),'confusion_matrix':cm.tolist(),'threshold':.5,'warnings':warnings}

def from_csv(path):
    with open(path,newline='') as f:rows=list(csv.DictReader(f))
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate prediction IDs')
    y=np.array([int(r['label_true']) for r in rows]);p=np.array([[float(r['score0']),float(r['score1'])] for r in rows])
    if any(int(r['label_pred'])!=int(p[i,1]>.5) for i,r in enumerate(rows)):raise ValueError('Prediction threshold mismatch')
    return metrics(y,p),rows,y,p

def better(auc,mse,best_auc,best_mse,tol=1e-6):
    return auc>best_auc+tol or (abs(auc-best_auc)<=tol and mse<best_mse)
