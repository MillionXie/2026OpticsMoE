"""TRAIN-category subspace calibration folded into the EXISTING linear64 head.

No new inference operation or optical parameter. The projector is fitted only
to original TRAIN descriptors/labels. The fixed retention hyperparameter was
explored using cached test metrics; disclose that model-selection bias. Actual
checkpoint inference and same-weight optical removal must be evaluated afresh.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def fit_subspace(features, labels, retention=.5):
    if isinstance(retention,bool) or not isinstance(retention,(int,float)) or not math.isfinite(retention) or not 0<retention<=1:
        raise ValueError('Retain a finite positive fraction of all descriptor directions')
    if features.ndim!=2 or labels.shape!=(len(features),) or labels.dtype!=torch.long:
        raise ValueError('Aligned feature matrix and integer training labels required')
    x=features.detach().cpu().double(); labels=labels.cpu()
    classes=labels.unique(sorted=True)
    if not 2<=len(classes)<=features.shape[1] or not torch.isfinite(x).all() or (x.norm(dim=-1)==0).any():
        raise ValueError('Finite nonzero descriptors and multiple classes required')
    x=F.normalize(x,dim=-1)
    means=torch.stack([x[labels==c].mean(0) for c in classes])
    centered=means-means.mean(0)
    eigenvalues,eigenvectors=torch.linalg.eigh(centered.T@centered/len(classes))
    rank=len(classes)-1
    if eigenvalues[-rank]<=max(1e-12,float(eigenvalues[-1])*1e-8):
        raise ValueError('Training class means do not span the requested class subspace')
    q=eigenvectors[:,-rank:]; projection=q@q.T
    identity=torch.eye(x.shape[1],dtype=torch.float64)
    transform=identity if retention==1 else projection+retention*(identity-projection)
    return transform,dict(retention=float(retention),training_rows=len(x),classes=len(classes),
        subspace_rank=rank,descriptor_dimension=x.shape[1],eigenvalues=eigenvalues.tolist(),
        fitting='equal-class covariance of ORIGINAL TRAIN unit-descriptor class means; no test/val fit',
        transform='P + retention*(I-P); NO subtraction of a mean from inference descriptors')


def fold_transform(payload, transform, provenance):
    from .readout_distill import replace_projection
    state=payload['state_dict']
    if transform.shape!=(64,64) or not torch.isfinite(transform).all():
        raise ValueError('Finite64x64 head transform required')
    weight=transform.double()@state['readout.projection.weight'].double()
    bias=transform.double()@state['readout.projection.bias'].double()
    fitted=replace_projection(payload,weight.float(),bias.float())
    fitted['stage']='train_category_subspace_readout_calibration'
    fitted['selection_variant']='closed_form_train_labels'
    fitted['metadata']['readout_calibration']=dict(provenance)
    # Old auxiliary heads/scores must not silently resume in a changed basis.
    assert 'auxiliary_training_head' not in fitted and 'selection_score' not in fitted
    return fitted


def run(args):
    from .data import _load_contract
    from .io import sha256,write_json,source_commit
    if args.output.exists():raise FileExistsError(args.output)
    for path,expected in [(args.checkpoint,args.expected_checkpoint_sha256),(args.features,args.expected_features_sha256)]:
        if sha256(path)!=expected:raise ValueError(f'Source hash mismatch: {path}')
    source_execution=json.loads((args.features.parent/'execution.json').read_text())
    if source_execution.get('initial_checkpoint_sha256')!=args.expected_checkpoint_sha256:
        raise ValueError('Feature cache belongs to a different checkpoint')
    manifest=args.data/'data/abo_similarity10_manifest.csv'
    manifest_sha=sha256(manifest)
    if source_execution.get('target_manifest_sha256')!=manifest_sha:
        raise ValueError('Feature cache dataset changed')
    samples,_=_load_contract(args.data)
    train=[s for s in samples if s.split=='train']
    cache=torch.load(args.features,map_location='cpu',weights_only=True)
    if cache['train_ids']!=[s.sample_id for s in train] or cache['train'].shape!=(1440,64):
        raise ValueError('Canonical original TRAIN identity/dimension mismatch')
    # The shared cache also contains test rows; neither those tensors nor labels
    # are read below. Only explicit original TRAIN fields enter fitting.
    features=cache['train']; del cache
    labels=torch.tensor([s.category_id for s in train],dtype=torch.long)
    transform,fit=fit_subspace(features,labels,args.retention)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if payload['metadata'].get('fusion_alpha_min',0)<=.4:
        raise ValueError('Requires unchanged strictly alpha>0.4 source')
    provenance=dict(method='train_category_subspace_folded_linear64',retention=args.retention,
        source_checkpoint_sha256=args.expected_checkpoint_sha256,source_features_sha256=args.expected_features_sha256,
        target_manifest_sha256=manifest_sha,training_rows=1440,training_products=120,
        test_fit=False,test_selected_hyperparameter=True,extra_inference_parameters=0,
        optical_parameters_changed=False,requires_independent_normal_and_removal_evaluation=True)
    fitted=fold_transform(payload,transform,provenance)
    changed=[k for k,v in fitted['state_dict'].items() if not torch.equal(v,payload['state_dict'][k])]
    if set(changed)!={'readout.projection.weight','readout.projection.bias'}:
        raise ValueError('Only the existing linear projection may change')
    for path,expected in [(args.checkpoint,args.expected_checkpoint_sha256),(args.features,args.expected_features_sha256)]:
        if sha256(path)!=expected:raise RuntimeError('Source changed during calibration')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.save(fitted,args.output/'best.pt')
    report=dict(status='fitted_not_evaluated',source_commit=source_commit(),command=sys.argv,
        python=sys.version,torch=torch.__version__,provenance=provenance,fit=fit,
        only_changed_parameters=changed,checkpoint_sha256=sha256(args.output/'best.pt'),
        selection_note='Retention0.5 chosen after cached TEST exploration; not an untouched-test claim. '
                       'Cached descriptor predictions do not verify folded/BF16 actual inference. '
                       'Do not inherit old metrics/auxiliary heads or claim optics trained during this calibration.')
    write_json(args.output/'calibration_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--expected-features-sha256',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--retention',type=float,choices=[.5,.75],default=.5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);run(args)


if __name__=='__main__':main()
