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
