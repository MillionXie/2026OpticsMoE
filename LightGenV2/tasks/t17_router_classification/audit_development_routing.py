"""Routing diagnostic only: fixed pair outputs and a true-label oracle, never LG accuracy."""
import argparse,itertools,json,os
from pathlib import Path
import numpy as np
import torch
from .model_four import FourRouterClassification
from .train import encode,write
from .test_exposed_data import load_raw_splits,metrics_from_scores
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


@torch.no_grad()
def score_model(model,x,batch):
    return torch.cat([model(x[i:i+batch])['probabilities'].cpu() for i in range(0,len(x),batch)]).numpy()


def diagnostic_label_selected_pair(scores,labels):
    """Counterfactual equal-power-pair diagnostic, not a realizable inference rule."""
    n,pairs,classes=scores.shape
    if len(labels)!=n or pairs!=6 or np.any(labels<0) or np.any(labels>=classes):
        raise ValueError('Expected N x 6 x C scores and N valid labels')
    correct=scores[np.arange(n)[:,None],np.arange(pairs)[None,:],labels[:,None]]
    other=scores.copy();other[np.arange(n)[:,None],np.arange(pairs)[None,:],labels[:,None]]=-np.inf
    choice=(correct-other.max(2)).argmax(1)
    return choice,scores[np.arange(n),choice]


def main():
    p=argparse.ArgumentParser()
    for name in ('profile','data-root','development-selection','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();gpu=require_non_a100_gpu();torch.set_num_threads(4);cfg=json.loads(a.profile.read_text());a.out.mkdir(parents=True,exist_ok=False)
    raw,identity=load_raw_splits(a.data_root,cfg,('test',));data=raw['test'];x=encode(data['images'],'four_top2').cuda()
    ck=torch.load(a.development_selection/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda().eval();model.load_state_dict(ck['model'])
    actual=score_model(model,x,cfg['batch']);original_route=model.route;pairs=list(itertools.combinations(range(4),2));scores=[];rows=[]
    for pair in pairs:
        power=torch.zeros(1,4,device='cuda');power[0,list(pair)]=.5
        model.route=lambda amplitude,power=power:(power.expand(len(amplitude),-1),None)
        result=score_model(model,x,cfg['batch']);scores.append(result)
        rows.append(dict(pair=list(pair),metrics=metrics_from_scores(data['labels'],result)))
    model.route=original_route
    all_scores=np.stack(scores,1);labels=data['labels'];choice,oracle=diagnostic_label_selected_pair(all_scores,labels)
    oracle_metrics=metrics_from_scores(labels,oracle)
    np.savez_compressed(a.out/'diagnostic_pair_outputs.npz',ids=data['ids'],labels=labels,actual_scores=actual,pairs=np.array(pairs),pair_scores=all_scores,TRUE_LABEL_ORACLE_pair=choice)
    result=dict(role='GROUND_TRUTH_ROUTING_ORACLE_DIAGNOSTIC_ONLY',independent_test_claim=False,not_an_inference_model=True,
                true_labels_used_to_select_pair=True,not_a_bound_on_variable_power_routing=True,
                source_checkpoint_sha256=sha256(a.development_selection/'best_checkpoint.pt'),
                source_commit=os.environ.get('T17_SOURCE_COMMIT'),gpu_uuid=gpu,data=identity,
                actual_optical_router=metrics_from_scores(labels,actual),fixed_equal_power_pairs=rows,
                TRUE_LABEL_ORACLE_NOT_LG_ACCURACY=oracle_metrics,oracle_pair_fraction=np.bincount(choice,minlength=6).astype(float).tolist())
    result['oracle_pair_fraction']=(np.bincount(choice,minlength=6)/len(choice)).tolist()
    write(a.out/'result.json',result);print(json.dumps({'actual':result['actual_optical_router']['accuracy'],'best_fixed_pair':max(r['metrics']['accuracy'] for r in rows),'GT_oracle_DIAGNOSTIC':oracle_metrics['accuracy']}))


if __name__=='__main__':main()
