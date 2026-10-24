"""One predeclared64-dimensional TRAIN-only, unlabeled ABO descriptor reference.

It matches the existing learned projection's dimensionality without SKU
supervision. PCA is a fitted preprocessing transform, not a claim of zero-data
or pretrained cross-modal inference. No dimensionality/whitening search occurs.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from .evaluate_retrieval_budget import load_feature_cache
from .frozen_resnet18 import FrozenResNet18,encoder_state_sha256,file_sha256
from .other_data import abo_data
from .retrieval import exact_sku_ranking
from .continue_lgvq50 import structured_sha256
from .train_easy100 import TASK,require_published_source,git,write_json,save_checkpoint


def fit_projection(train, dimensions=64):
    if train.ndim!=2 or not 0<dimensions<=min(train.shape):raise ValueError('PCA shape/dimension differs')
    normalized=F.normalize(train.float(),dim=-1)
    mean=normalized.double().mean(0)
    _,singular,vh=torch.linalg.svd(normalized.double()-mean,full_matrices=False)
    return dict(mean=mean.float(),components=vh[:dimensions].float(),singular_values=singular[:dimensions].float())


def project(features, transform):
    return F.normalize((F.normalize(features.float(),dim=-1)-transform['mean'])@transform['components'].T,dim=-1)


def main(args):
    require_published_source()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('This readout diagnostic is CPU-only')
    torch.set_num_threads(4)
    run=args.run_dir.resolve()
    if not run.is_relative_to((TASK/'runs/simulation').resolve()):raise ValueError('Run outside task')
    run.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='preflight',pid=os.getpid(),started_unix=time.time(),git_commit=git('rev-parse','HEAD'),
        command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],dimensions=64,whiten=False,
        fitting_data='original1600 TRAIN views only; no SKU supervision',optimizer_steps=0,
        historical_TEST_observed=True,target_score_used_to_choose_dimension=False,
        source_SHA={name:file_sha256(TASK/name) for name in ('retrieval_pca_reference.py','evaluate_retrieval_budget.py','retrieval.py')})
    write_json(run/'run_manifest.json',manifest)
    try:
        train,test,identity=abo_data();encoder=FrozenResNet18();enc_sha=encoder_state_sha256(encoder)
        base=TASK/'runs/simulation/retrieval_fixed_budget_cpu_20261024/cache/abo200'
        a,ma=load_feature_cache(base/'train',data_identity=identity,encoder_hash=enc_sha)
        b,mb=load_feature_cache(base/'test',data_identity=identity,encoder_hash=enc_sha)
        if (a['sample_ids']!=[r['sample_id'] for r in train.rows] or b['sample_ids']!=[r['sample_id'] for r in test.rows]):
            raise RuntimeError('Original cache order differs')
        transform=fit_projection(a['features'],64)
        scores=project(b['features'],transform)@project(a['features'],transform).T
        ranking=exact_sku_ranking(scores,b['skus'],a['skus'])
        save_checkpoint(run/'pca_transform.pt',transform)
        write_json(run/'fixed_weight_predictions.json',dict(query_ids=b['sample_ids'],candidate_ids=a['sample_ids'],**ranking))
        write_json(run/'result.json',dict(status='complete',metrics={k:v for k,v in ranking.items() if isinstance(v,(int,float))},
            dimensions=64,whiten=False,supervised_optimizer_steps=0,PCA_fitted_on_TRAIN=True,
            transform_SHA=file_sha256(run/'pca_transform.pt'),transform_content_SHA=structured_sha256(transform),
            train_cache_binding=ma['cache_binding_sha256'],test_cache_binding=mb['cache_binding_sha256'],
            test_metric_passes=1,TEST_vectors_used_to_fit_projection=False))
        manifest.update(status='complete',data_identity=identity,encoder_SHA=enc_sha,
            pretrained_provenance=encoder.pretrained_provenance)
    except BaseException as exc:manifest.update(status='failed',error=repr(exc));raise
    finally:manifest['finished_unix']=time.time();write_json(run/'run_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path)
    main(parser.parse_args())
