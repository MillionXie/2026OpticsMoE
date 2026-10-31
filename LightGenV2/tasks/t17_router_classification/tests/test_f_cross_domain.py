import json
import numpy as np
import pytest
from LightGenV2.tasks.t17_router_classification.f_cross_domain import select_source_domain,validate_domain_teacher_cache
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256
import torch
from LightGenV2.tasks.t17_router_classification.train_figure_revision import evaluate


@pytest.mark.parametrize('domain,number',[('RGB',0),('SAR',1)])
def test_single_domain_training_selection_retains_both_test_domains_without_mutation(domain,number):
    raw={s:dict(images=np.zeros((4,2,2,3),dtype=np.uint8),ids=np.array(['a','b','c','d']),labels=np.array([0,0,1,1]),domains=np.array([0,1,0,1])) for s in ('train','val','test')}
    selected=select_source_domain(raw,domain)
    assert all(len(raw[s]['labels'])==4 for s in raw)
    assert (selected['train']['domains']==number).all() and (selected['val']['domains']==number).all()
    assert np.array_equal(selected['test']['ids'],raw['test']['ids']) and len(selected['test']['ids'])==4
    selected['train']['labels'][0]=9;assert raw['train']['labels'][number]==0


def test_domain_teacher_rejects_other_domain_or_corrupt_cache(tmp_path):
    train=dict(ids=np.array(['a','b']),labels=np.array([0,1]),domains=np.array([0,0]));cfg=dict(training_domain='RGB',data={},num_classes=10)
    np.savez(tmp_path/'teacher_logits_train.npz',**train,logits=np.zeros((2,8,10),dtype=np.float32));(tmp_path/'best_checkpoint.pt').write_bytes(b'identity')
    status=dict(state='complete',opposite_domain_used=False,checkpoint_sha256=sha256(tmp_path/'best_checkpoint.pt'));meta=dict(training_domain='RGB',original_data={},opposite_domain_used=False)
    (tmp_path/'status.json').write_text(json.dumps(status));(tmp_path/'metadata.json').write_text(json.dumps(meta));(tmp_path/'cache_manifest.json').write_text(json.dumps({'teacher_logits_train.npz':sha256(tmp_path/'teacher_logits_train.npz')}))
    assert validate_domain_teacher_cache(dict(train=train),cfg,tmp_path)[0].shape==(2,8,10)
    with pytest.raises(ValueError,match='different-domain'):validate_domain_teacher_cache(dict(train=train),dict(cfg,training_domain='SAR'),tmp_path)
    meta['opposite_domain_used']=True;(tmp_path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='opposite domain'):validate_domain_teacher_cache(dict(train=train),cfg,tmp_path)
    meta['opposite_domain_used']=False;(tmp_path/'metadata.json').write_text(json.dumps(meta));(tmp_path/'teacher_logits_train.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA'):validate_domain_teacher_cache(dict(train=train),cfg,tmp_path)


def test_single_domain_evaluation_does_not_create_empty_opposite_domain_metrics():
    class Toy(torch.nn.Module):
        def forward(self,x):return dict(probabilities=torch.tensor([[.8,.2]]).repeat(len(x),1))
    metrics,_=evaluate(Toy(),torch.zeros(2,1),torch.zeros(2,dtype=torch.long),2,np.array([0,0]))
    assert set(metrics['domains'])=={'RGB'} and metrics['domain_mean_accuracy']==1.
