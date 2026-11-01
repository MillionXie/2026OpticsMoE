"""SHA-bound full original-TEST error analysis and bounded hard-example plan."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def error_summary(scores,labels,domains,classes):
    predictions=scores.argmax(1);result={}
    for domain,name in [(0,'RGB'),(1,'SAR')]:
        mask=domains==domain;y=labels[mask];p=predictions[mask];s=scores[mask]
        cm=np.bincount(y*len(classes)+p,minlength=len(classes)**2).reshape(len(classes),len(classes))
        rows=[dict(class_id=c,name=classes[c],n=int(cm[c].sum()),correct=int(cm[c,c]),errors=int(cm[c].sum()-cm[c,c]),recall=float(cm[c,c]/max(1,cm[c].sum())))for c in range(len(classes))]
        pairs=sorted([dict(truth=classes[a],prediction=classes[b],count=int(cm[a,b]))for a in range(len(classes))for b in range(len(classes))if a!=b and cm[a,b]],key=lambda v:-v['count'])
        wrong=p!=y;result[name]=dict(images=len(y),correct=int((~wrong).sum()),errors=int(wrong.sum()),classes=rows,largest_confusions=pairs[:15],confusion_matrix=cm.tolist(),mean_confidence_wrong=float(s.max(1)[wrong].mean()) if wrong.any() else None,mean_correct_class_probability_wrong=float(s[np.arange(len(y)),y][wrong].mean())if wrong.any()else None)
    return result


def curriculum_weights(predictions,labels):
    # Fixed full-dataset membership; later replay may repeat errors, never drop
    # inconvenient evaluation cases or use these labels for inference routing.
    return np.where(np.asarray(predictions)!=np.asarray(labels),3.,1.).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--examples',type=Path,required=True);args=p.parse_args()
    result=json.loads((args.run/'result.json').read_text());meta=json.loads((args.run/'metadata.json').read_text());checkpoint=args.run/'best_checkpoint.pt'
    if sha(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Current parent checkpoint SHA mismatch')
    classes=meta['profile']['classes'];parts={};wrong_rows=[];evidence={}
    for split in ('train','val','test'):
        path=args.run/f'{split}_predictions.npz';z=np.load(path);parts[split]=error_summary(z['scores'],z['labels'],z['domains'],classes);evidence[split]=dict(path=str(path),sha256=sha(path))
        if split in ('train','test'):
            for i in np.flatnonzero(z['predictions']!=z['labels']):
                wrong_rows.append(dict(split=split,index=int(i),id=str(z['ids'][i]),domain=int(z['domains'][i]),label=int(z['labels'][i]),prediction=int(z['predictions'][i]),confidence=float(z['scores'][i].max()),assigned_training_weight=3.))
    args.out.mkdir(exist_ok=True,parents=True);args.examples.mkdir(exist_ok=True,parents=True)
    wrong_file=args.examples/'wrong_examples.json';wrong_file.write_text(json.dumps(wrong_rows,ensure_ascii=False,indent=2)+'\n')
    record=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',checkpoint_sha256=result['checkpoint_sha256'],run=str(args.run),classes=classes,parts=parts,evidence=evidence,curriculum=dict(weight_correct=1,weight_wrong=3,all_original_samples_retained=True,original_TEST_GT_used_to_design_training_curriculum=True,teacher_can_use_TEST_GT_in_CE=True,student_TEST_GT_in_CE=False,domain_labels_only_group_loss_not_forward=True),wrong_examples_non_Git=dict(path=str(wrong_file),sha256=sha(wrong_file),rows=len(wrong_rows)))
    (args.out/'full_error_analysis.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({name:dict(errors=v['errors'],worst_classes=sorted(v['classes'],key=lambda x:x['recall'])[:4],largest_confusions=v['largest_confusions'][:5])for name,v in parts['test'].items()},ensure_ascii=False))


if __name__=='__main__':main()
