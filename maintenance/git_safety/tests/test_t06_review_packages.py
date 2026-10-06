import json
import pytest
from maintenance.git_safety.audit_t06_review_packages import inspect, sha


def package(tmp_path):
    (tmp_path/'weights').mkdir()
    (tmp_path/'inputs').mkdir()
    (tmp_path/'weights/best_checkpoint.pt').write_bytes(b'not-a-model')
    (tmp_path/'inputs/f.pt').write_bytes(b'not-features')
    field={'file':'inputs/f.pt','sha256':sha(tmp_path/'inputs/f.pt'),
           'valid':[True,False], 'sample_ids':['one','one'], 'targets':[1,1]}
    release={'fields':[field], 'checkpoint_sha256':sha(tmp_path/'weights/best_checkpoint.pt'),
             'test_videos_in_package':1, 'simulation_metrics':{'srcc':.8}}
    (tmp_path/'release.json').write_text(json.dumps(release))
    manifest={p:sha(tmp_path/p) for p in ('release.json','weights/best_checkpoint.pt','inputs/f.pt')}
    (tmp_path/'SHA256.json').write_text(json.dumps(manifest))
    return manifest


def test_integrity_without_loading_model_and_padding_excluded(tmp_path):
    package(tmp_path)
    result=inspect(tmp_path)
    assert result['integrity_passed'] and result['valid_videos']==1
    assert not result['model_evaluated'] and result['mutations']==[]


def test_changed_asset_fails(tmp_path):
    package(tmp_path)
    (tmp_path/'inputs/f.pt').write_bytes(b'changed')
    assert not inspect(tmp_path)['integrity_passed']


def test_unbound_checkpoint_fails(tmp_path):
    manifest=package(tmp_path)
    del manifest['weights/best_checkpoint.pt']
    (tmp_path/'SHA256.json').write_text(json.dumps(manifest))
    assert not inspect(tmp_path)['integrity_passed']


def test_external_manifest_path_rejected(tmp_path):
    manifest=package(tmp_path)
    manifest['../outside.py']='0'*64
    (tmp_path/'SHA256.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Unsafe'):
        inspect(tmp_path)
