"""Read-only TRAIN dropout sensitivity of the native-audited original head.

Analytic independent Bernoulli variance and fixed-seed TRAIN probe, no fitting,
TEST inspection, SDK access, new inference layer or modification of a PT.
"""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_dvp_layerwise_20261019 as m
from lab_caltech20_dvp_completed_readout_20261021 import native_probability
from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry


def audit(parent,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();torch.set_num_threads(2);torch.manual_seed(73)
    report=m.read(parent/'report.json');assert report['full_physical'] and report['strict_original_CPU'];cfg=m.read(Path(__file__).with_name('lab_caltech20_tucam_fixed_L4_profile_20261023.json'));protocol=m.read(parent/'protocol.json')
    pt=parent/'best.pt';assert m.sha(pt)==report['results']['best']['PT_SHA256'];model,_=m.model_for(pt,m.sha(pt),4,m.ROOT/cfg['L4_reference'])
    p=parent/'train_features.npz';assert m.sha(p)==protocol['feature_SHA256']['train']
    with np.load(p) as z:x=torch.from_numpy(z['x'].copy()).float();y=torch.from_numpy(z['y'].copy()).long();ids=z['ids'].copy()
    assert len(x)==1600 and x.shape[1]==256
    w=model.linear.weight.detach();b=model.linear.bias.detach();prob=native_probability(torch,x,w,b)
    with np.load(parent/'best_train_predictions.npz') as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y.numpy(),z['labels']) and float((prob-torch.from_numpy(z['probabilities'])).abs().max())<=3e-6
    z=F.linear(x,w,b);wrong=z.masked_fill(F.one_hot(y,20).bool(),-torch.inf).argmax(1);margin=z[torch.arange(len(y)),y]-z[torch.arange(len(y)),wrong];difference=w[y]-w[wrong];good=margin>0;drop=.03
    sigma=(drop/(1-drop)*(difference*x).square().sum(1)).sqrt();ratio=sigma[good]/margin[good].clamp_min(1e-12)
    mean,root,_,_,floor=train_conditioned_geometry(x.numpy());inverse=torch.linalg.inv(root);xx=(x.double()-mean)@root;assert float((((xx@inverse+mean)-x.double()).abs().max()))<1e-8
    views={'clean':x,'independent_raw_feature_dropout':F.dropout(x,p=drop,training=True),'TRAIN_conditioned_dropout_restored_to_original_features':(F.dropout(xx,p=drop,training=True)@inverse+mean).float()};rows={}
    for name,view in views.items():
        logits=F.linear(view,w,b);bad=logits.masked_fill(F.one_hot(y,20).bool(),-torch.inf).max(1).values
        rows[name]=dict(TRAIN_correct=int((logits.argmax(1)==y).sum()),TRAIN_CE=float(F.cross_entropy(logits,y)),TRAIN_squared_hinge=float((1+bad-logits[torch.arange(len(y)),y]).clamp_min(0).square().mean()))
    result=dict(passed=True,TRAIN=1600,TEST_used=False,VALID=0,fit_steps=0,PT_written=False,SDK_used=False,GPU_used=False,source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(pt),native_report_SHA256=m.sha(parent/'report.json'),TRAIN_feature_SHA256=m.sha(p),dropout=drop,variance_floor=floor,TRAIN_view_probe=rows,correct_TRAIN_margin_median=float(margin[good].median()),independent_dropout_margin_sigma_median=float(sigma[good].median()),sigma_to_positive_margin_quantiles=torch.quantile(ratio,torch.tensor([.1,.5,.9])).tolist(),interpretation='Read-only augmentation sensitivity, not a new accuracy result or proof of optical information loss. Restored conditioning is a TRAIN transform only; inference remains unchanged.')
    m.save(out,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.parent,a.out)
