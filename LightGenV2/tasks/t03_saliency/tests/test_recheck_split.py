from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader, SequentialSampler

from LightGenV2.tasks.t03_saliency import recheck_aligned as recheck


@pytest.mark.parametrize('split,count,index',[('train',10000,0),('test',5000,1)])
def test_full_split_recheck_is_unaugmented_sequential_and_spawned(monkeypatch,split,count,index):
    bundle=SimpleNamespace(train_records=list(range(10000)),validation_records=list(range(5000)))
    calls=[]
    loaders=[DataLoader(bundle.train_records,num_workers=1),DataLoader(bundle.validation_records,num_workers=1)]
    def build(b,s,training):
        assert b is bundle
        calls.append(training)
        return loaders
    monkeypatch.setattr(recheck.legacy,'build_loaders',build)
    loader,expected=recheck.recheck_loader(bundle,None,split)
    assert calls==[False] and expected==count and loader is loaders[index]
    assert isinstance(loader.sampler,SequentialSampler)
    assert loader.multiprocessing_context.get_start_method()=='spawn'


@pytest.mark.parametrize('split,train,test',[('train',9999,5000),('test',10000,4999),('val',10000,5000)])
def test_recheck_rejects_incomplete_or_ambiguous_split(split,train,test):
    bundle=SimpleNamespace(train_records=range(train),validation_records=range(test))
    with pytest.raises((ValueError,RuntimeError)):
        recheck.recheck_loader(bundle,None,split)


def test_optical_checkpoint_view_is_explicit_and_never_mutates_payload():
    p={'core':{'value':1},'saliency_head':{'value':2},'weight_kind':'live',
       'ema_state':{'core':{'value':3},'head':{'value':4}}}
    assert recheck.optical_checkpoint_states(p)==(p['core'],p['saliency_head'],'live')
    assert recheck.optical_checkpoint_states(p,True)==(p['ema_state']['core'],p['ema_state']['head'],'ema')
    assert p['core']=={'value':1} and p['saliency_head']=={'value':2}
    p['weight_kind']='ema'
    assert recheck.optical_checkpoint_states(p)[2]=='ema'


@pytest.mark.parametrize('ema',[None,{}, {'core':{}}, {'core':{},'head':{},'extra':{}}])
def test_missing_ema_never_silently_evaluates_live_weights(ema):
    p={'core':{},'saliency_head':{},'ema_state':ema}
    with pytest.raises(ValueError,match='refusing live fallback'):
        recheck.optical_checkpoint_states(p,True)
