"""Explicit original-TEST development selection and error analysis, never held-out reporting."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import torch
from .model_four import FourRouterClassification,ccd_grid_energies
from .model_linear_ccd import LinearD2NNCCD
from .train import encode,write
from .train_optical_four_ccd import route_statistics
from .test_exposed_data import load_raw_splits,metrics_from_scores
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


@torch.no_grad()
def evaluate_original_test(model,x,labels,batch,is_lg):
    probabilities=[];routes=[];raw=[];capture=[]
    model.eval()
    for i in range(0,len(labels),batch):
        out=model(x[i:i+batch],return_dense_route=True,return_debug=True) if is_lg else model(x[i:i+batch])
        probabilities.append(out['probabilities'].cpu().numpy())
        if is_lg:
            routes.append(out['route_power'].cpu().numpy());capture.append(out['router_capture'].cpu().numpy())
            e=ccd_grid_energies(out['final_ccd'][:,20:498,20:498],len(out['logits'][0]))
            raw.append(((e+1e-12)/(e.sum(1,keepdim=True)+e.shape[1]*1e-12)).cpu().numpy())
    p=np.concatenate(probabilities);metrics=metrics_from_scores(labels,p)
    q=np.concatenate(routes) if routes else None
    if is_lg:
        metrics['router']=route_statistics(q);metrics['router_capture_mean']=float(np.concatenate(capture).mean())
        metrics['raw_CCD_before_second_OEO_DIAGNOSTIC_ONLY']=metrics_from_scores(labels,np.concatenate(raw))
    return metrics,p,q


def main():
    ap=argparse.ArgumentParser()
    for name in ('profile','data-root','base','refined','d2nn','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(a.profile.read_text())
    if json.loads((a.refined/'status.json').read_text())['state']!='complete':raise ValueError('wait for prescribed validation refinement to finish')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    raw,identity=load_raw_splits(a.data_root,cfg,('test',));test=raw['test'];x=encode(test['images'],'four_top2').cuda();classes=cfg['num_classes']
    reference_metadata=json.loads((a.d2nn/'metadata.json').read_text())
    reference_archive=reference_metadata.get('data',{}).get('archive_sha256',reference_metadata.get('data_sha256'))
    if reference_archive!=identity['archive_sha256']:raise ValueError('D2NN reference dataset mismatch')
    write(a.out/'exposure.json',dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',original_test_used_for_checkpoint_selection=True,
          original_test_used_for_error_diagnosis=True,original_test_in_teacher_training=False,original_test_in_student_training=False,
          independent_test_claim=False,data=identity,gpu_uuid=gpu,command=os.sys.argv,source_commit=os.environ.get('T17_SOURCE_COMMIT')))
    specifications=[('base_best',a.base/'best_checkpoint.pt','model',True),('base_last',a.base/'last_checkpoint.pt','model',True),
                    ('refined_best',a.refined/'best_checkpoint.pt','model',True),('refined_last',a.refined/'last_checkpoint.pt','model',True),
                    ('refined_last_ema',a.refined/'last_checkpoint.pt','ema',True),
                    ('d2nn_val_best',a.d2nn/'best_checkpoint.pt','model',False),('d2nn_last',a.d2nn/'last_checkpoint.pt','model',False)]
    rows=[]
    for tag,source,key,is_lg in specifications:
        original_sha=sha256(source);ck=torch.load(source,map_location='cpu',weights_only=False)
        if sha256(source)!=original_sha:raise RuntimeError('source checkpoint changed')
        state=ck[key]
        model=(FourRouterClassification('optical','ccd_grid',num_classes=classes) if is_lg else LinearD2NNCCD(num_classes=classes)).cuda()
        model.load_state_dict(state);assert sum(p.numel() for p in model.parameters())==(479364 if is_lg else 685452)
        snapshot=a.out/'candidates'/tag;snapshot.mkdir(parents=True)
        torch.save(dict(model=state,source_checkpoint=str(source),source_checkpoint_sha256=original_sha,source_state_key=key,source_epoch=ck['epoch'],
                        role='original_TEST_dev_selection_candidate'),snapshot/'best_checkpoint.pt')
        metrics,p,q=evaluate_original_test(model,x,test['labels'],cfg['batch'],is_lg)
        np.savez_compressed(snapshot/'original_test_dev_predictions.npz',ids=test['ids'],labels=test['labels'],scores=p,predictions=p.argmax(1),
                            **({'route_power':q} if q is not None else {}))
        row=dict(candidate=tag,is_LG=is_lg,source=str(source),source_checkpoint_sha256=original_sha,snapshot_sha256=sha256(snapshot/'best_checkpoint.pt'),
                 state_key=key,source_epoch=ck['epoch'],original_test_dev=metrics)
        rows.append(row);write(snapshot/'metrics.json',row);print(tag,metrics['accuracy'],flush=True)
        del model;torch.cuda.empty_cache()
    key=lambda r:(r['original_test_dev']['accuracy'],r['original_test_dev']['balanced_accuracy'],-r['original_test_dev']['cross_entropy'])
    selected=max((r for r in rows if r['is_LG']),key=key);reference=max((r for r in rows if not r['is_LG']),key=key)
    chosen_path=a.out/'candidates'/selected['candidate'];chosen=torch.load(chosen_path/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    chosen.update(selection='Original TEST development accuracy then macro recall then CE',role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT')
    torch.save(chosen,a.out/'best_checkpoint.pt')
    z=np.load(chosen_path/'original_test_dev_predictions.npz',allow_pickle=False);p=z['scores'];pred=z['predictions'];y=test['labels']
    margins=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
    wrong=np.flatnonzero(pred!=y);hard=wrong[np.argsort(p[wrong,y[wrong]])][:32]
    examples=a.out/'error_examples';examples.mkdir();cards=[];review=[]
    for i in hard:
        name=f'{i:05d}.png';Image.fromarray(test['images'][i]).save(examples/name)
        item=dict(sample_id=str(test['ids'][i]),index=int(i),original_label=int(y[i]),prediction=int(pred[i]),
                  original_class=identity['classes'][y[i]],predicted_class=identity['classes'][pred[i]],
                  correct_class_probability=float(p[i,y[i]]),predicted_probability=float(p[i,pred[i]]),margin=float(margins[i]),image='error_examples/'+name)
        review.append(item);cards.append('<figure><img width="150" src="'+item['image']+'"><figcaption>'+html.escape(item['sample_id'])+'<br>GT '+html.escape(item['original_class'])+'<br>Pred '+html.escape(item['predicted_class'])+'</figcaption></figure>')
    write(a.out/'error_review.json',dict(label_changes_made=False,removed_samples=[],items=review,
          note='These are model errors for review, not established annotation errors.'))
    (a.out/'error_review.html').write_text('<meta charset="utf-8"><h1>Original TEST development errors</h1><p>TEST-exposed selection; not independent held-out evaluation. Original labels/images retained.</p><div style="display:flex;flex-wrap:wrap">'+''.join(cards)+'</div>',encoding='utf-8')
    fixed=next(r for r in rows if r['candidate']=='d2nn_val_best')
    original_reference=json.loads((a.d2nn/'result.json').read_text())
    assert abs(fixed['original_test_dev']['accuracy']-original_reference['test']['accuracy'])<1e-6
    result=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',independent_test_claim=False,candidates=rows,
                selected_LG=selected,selected_D2NN_development_reference=reference,frozen_validation_selected_D2NN=fixed,
                gain_over_frozen_D2NN_pp=100*(selected['original_test_dev']['accuracy']-fixed['original_test_dev']['accuracy']),
                gain_over_test_selected_D2NN_pp=100*(selected['original_test_dev']['accuracy']-reference['original_test_dev']['accuracy']),
                original_test_selection_candidate_count=len(rows),teacher_fallback_recommended=100*(selected['original_test_dev']['accuracy']-reference['original_test_dev']['accuracy'])<20)
    write(a.out/'result.json',result);write(a.out/'status.json',dict(state='complete',selected_LG=selected['candidate'],role=result['role']))
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidates','selected_LG','selected_D2NN_development_reference','frozen_validation_selected_D2NN')}),flush=True)


if __name__=='__main__':main()
