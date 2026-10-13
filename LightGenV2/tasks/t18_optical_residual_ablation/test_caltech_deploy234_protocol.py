"""Synthetic checks for refusal/claim ordering; no model forward or real data."""
from pathlib import Path
from types import SimpleNamespace
import json,os
import pytest
import caltech_deploy234 as t
import caltech_deploy234_finalize as f

def test_partial_lock_creates_nothing(tmp_path,monkeypatch):
    out=tmp_path/'final';a=SimpleNamespace(base=tmp_path/'base',out=out,data=tmp_path/'data.npz')
    def rejected(_):raise AssertionError('six fits not complete')
    monkeypatch.setattr(f,'audit_all',rejected)
    with pytest.raises(AssertionError):f.lock(a)
    assert not out.exists()

def test_partial_lock_directory_is_not_retried(tmp_path,monkeypatch):
    out=tmp_path/'final';out.mkdir();(out/'partial_marker').write_text('keep')
    monkeypatch.setattr(f,'audit_all',lambda _:pytest.fail('must refuse before audit'))
    with pytest.raises(AssertionError):f.lock(SimpleNamespace(out=out))
    assert (out/'partial_marker').read_text()=='keep'

def fake_worker(tmp_path,monkeypatch):
    out=tmp_path/'final';base=tmp_path/'fit';key='L4_rho03';(out/key).mkdir(parents=True);(base/key).mkdir(parents=True)
    ck=base/key/'best_checkpoint.pt';ck.write_bytes(b'not loaded')
    (out/key/'selection_lock.json').write_text(json.dumps(dict(checkpoint_sha256=t.sha(ck),data_sha256='frozen')))
    (base/key/'fit_result.json').write_text(json.dumps(dict(checkpoint_sha256=t.sha(ck))))
    monkeypatch.setattr(f,'validate_lock',lambda _:dict(all_six_fit_audited=True))
    monkeypatch.setattr(t.gpu_budget,'configure',lambda *args,**kwargs:None)
    gpu=t.ALLOWED[2];monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu)
    return SimpleNamespace(base=base,out=out,key=key,data=tmp_path/'data.npz',gpu=gpu)

def test_failed_loader_leaves_claim_and_refuses_retry(tmp_path,monkeypatch):
    a=fake_worker(tmp_path,monkeypatch);calls=[]
    def loader(_):
        assert (a.out/a.key/'test_started.json').exists()
        calls.append('after_claim');raise RuntimeError('synthetic loader failure')
    monkeypatch.setattr(t,'manifest',loader)
    with pytest.raises(RuntimeError):f.test_worker(a)
    assert calls==['after_claim'] and (a.out/a.key/'test_started.json').exists()
    with pytest.raises(FileExistsError):f.test_worker(a)
    assert calls==['after_claim']

def test_exclusive_claim_preserves_original(tmp_path):
    path=tmp_path/'claim.json';t.save(path,dict(checkpoint='original'),exclusive=True)
    with pytest.raises(FileExistsError):t.save(path,dict(checkpoint='changed'),exclusive=True)
    assert t.read(path)==dict(checkpoint='original')