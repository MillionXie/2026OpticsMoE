import random
import sys
import pytest
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.randomness import training_seed, epoch_random_streams


@pytest.mark.parametrize('value', [-1, 2**32, 1.5, True, '1.5', 'bad'])
def test_invalid_seed_rejected(value):
    with pytest.raises(ValueError):training_seed(value)


def test_default_streams_match_legacy_exactly():
    for epoch in (1, 7, 16):
        actual=epoch_random_streams(42,epoch)
        legacy=(random.Random(42+epoch),random.Random(19042+epoch))
        for a,b in zip(actual,legacy):
            assert [a.random() for _ in range(80)]==[b.random() for _ in range(80)]
    assert training_seed('123')==123
    assert training_seed(2**32-1)==2**32-1


def test_seed_and_epoch_change_sequence_reproducibly():
    def draws(seed,epoch):
        a,b=epoch_random_streams(seed,epoch)
        return [a.random() for _ in range(30)],[b.random() for _ in range(30)]
    assert draws(123,1)==draws(123,1)
    assert draws(123,1)!=draws(42,1)
    assert draws(123,1)!=draws(123,2)
    for epoch in (0,1.5,True):
        with pytest.raises(ValueError):epoch_random_streams(42,epoch)


def test_queue_invalid_seed_fails_before_output(tmp_path,monkeypatch):
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone import generalization_queue
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['queue','--gpu','fake','--assets','a','--target','t','--checkpoint','c',
        '--output',str(out),'--seed','-1'])
    with pytest.raises(SystemExit) as exc:generalization_queue.main()
    assert exc.value.code==2 and not out.exists()


def test_seeded_sampler_keeps_product_coverage_and_split_contract():
    from types import SimpleNamespace
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.domain_data import epoch_batches
    samples=[]
    for domain in ('target','external'):
        for c in range(10):
            for p in range(6):
                for view in range(2):
                    samples.append(SimpleNamespace(category_id=c,product_id=f'{domain}-{c}-{p}'))
    def make(seed):
        rng,_=epoch_random_streams(seed,1)
        phase,batches,active=epoch_batches(samples,120,'mixed',1,0,6,rng)
        assert phase=='mixed' and len(active)==240
        assert {samples[i].product_id for b in batches for i in b}=={s.product_id for s in samples}
        assert all(len(b)==40 and sum(i<120 for i in b)==20 for b in batches)
        return batches
    assert make(123)==make(123)
    assert make(42)!=make(123)
