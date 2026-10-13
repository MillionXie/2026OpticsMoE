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

@pytest.mark.parametrize('depth,rho',[(d,r) for d in (2,3,4) for r in (0.,.3)])
def test_static_joint_export_dimensions_and_encoded_field(tmp_path,monkeypatch,depth,rho):
    import math,numpy as np,torch
    from PIL import Image
    import export_caltech_deploy234 as e
    from caltech_deploy234_model import DeployMoE
    torch.set_num_threads(2);torch.manual_seed(17);model=DeployMoE(depth,rho).eval()
    monkeypatch.setattr(model,'forward',lambda *args,**kwargs:pytest.fail('No extra model inference allowed'))
    checkpoint=tmp_path/'selected.pt';checkpoint.write_bytes(b'synthetic reference only')
    package=tmp_path/'deployment';manifest=e.export_selected(model,dict(depth=depth,residual_rho=rho),package,checkpoint,e.sha(checkpoint),[str(k) for k in range(20)],dict(synthetic='only'))
    assert manifest['phase_only_is_sufficient'] is False and manifest['model_forward_calls']==0
    assert all(e.sha(package/name)==identity for name,identity in manifest['files'].items())
    masks=np.load(package/'fixed_masks.npz');assert masks['detectors'].shape==(20,478,478) and masks['detectors'].sum(0).max()==1
    assert json.loads((package/'classes.json').read_text())['class_label']==list(range(20))
    amp=torch.stack([torch.zeros(224,224),torch.linspace(0,.5,224*224).reshape(224,224)])
    weights=torch.tensor([[.6,0.,.8,0.],[.6,0.,.8,0.]])
    with np.load(package/'amplitude_multipliers.npz') as maps:
        for stage in ['router']+[str(k) for k in range(depth)]:
            with torch.no_grad():plane=model.plane(stage).cpu()[20:498,20:498]
            assert np.array_equal(maps[stage],plane.abs().numpy())
            canonical=np.asarray(Image.open(package/f'phase_{stage}_canonical.bmp'))
            device=np.asarray(Image.open(package/f'phase_{stage}_native.bmp'))
            assert canonical.shape==(478,478) and device.shape==(1200,1920)
            expected=torch.floor(torch.remainder(torch.angle(plane),2*math.pi)/(2*math.pi)*256).clamp(0,255).to(torch.uint8).numpy()
            assert np.array_equal(canonical,expected)
            w=None if stage=='router' else weights
            a,p=e.encode_stage(model,amp,stage,w)
            with torch.no_grad():field,_,_=model.physical_fields(amp,stage,w)
            decoded=torch.from_numpy(a.astype(np.float32)/255)*torch.exp(2j*math.pi*torch.from_numpy(p.astype(np.float32)/256))[None]
            assert (decoded-field[:,20:498,20:498]).abs().max()<1e-6
            sample=tmp_path/('sample_'+stage);receipt=e.save_sample_bmps(model,amp,stage,sample,w)
            assert np.asarray(Image.open(sample/'amplitude_000000_canonical.bmp')).shape==(478,478)
            assert np.asarray(Image.open(sample/'amplitude_000001_native.bmp')).shape==(1080,1920)
            assert receipt['brightness'][0]['zero_fraction']==1. and receipt['brightness'][0]['active']['maximum']==0.
            assert receipt['brightness'][1]['extra_peak_normalization'] is False
            assert np.array_equal(np.asarray(Image.open(sample/'amplitude_000001_canonical.bmp')),a[1])
            assert not np.asarray(Image.open(sample/'amplitude_000000_native.bmp')).any()
