import json
from pathlib import Path
import pytest
from LightGenV2.tasks.t07_abo_image_retrieval import build_lab_package as pack


def fixture_files(tmp_path, monkeypatch):
    assets=tmp_path/'assets';data=tmp_path/'data';ref=tmp_path/'reference'
    (assets/'processor').mkdir(parents=True);data.mkdir();(ref/'verification').mkdir(parents=True)
    (assets/'processor/config.json').write_text('{}')
    (assets/'best.pt').write_bytes(b'WRONG old checkpoint')
    (ref/'best.pt').write_bytes(b'new verified checkpoint')
    (data/'photo.jpg').write_bytes(b'photo')
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'rows':[{'sample_id':'a','image_path':'photo.jpg','image_sha256':pack.digest(data/'photo.jpg')}]}))
    monkeypatch.setattr(pack,'BEST_83125',pack.digest(ref/'best.pt'))
    monkeypatch.setattr(pack,'PROTOCOL_83125',pack.digest(protocol))
    (ref/'verification/final_report.json').write_text(json.dumps({'checkpoint_sha256':pack.BEST_83125,
        'manifest_sha256':pack.PROTOCOL_83125,'metrics':{'normal':{'hit_at_1':.83125}}}))
    for name in ('execution.json','normal_predictions.csv','remove_optical_predictions.csv','weight_train_audit.json'):
        (ref/'verification'/name).write_text('{}')
    (ref/'final_report.json').write_text('{}')
    return assets,data,ref,protocol


def test_release_replaces_legacy_assets_and_keeps_current_data(tmp_path,monkeypatch):
    assets,data,ref,protocol=fixture_files(tmp_path,monkeypatch)
    files,generated=pack.enrolled_files(assets,data,ref,protocol)
    assert files['assets/best.pt']==ref/'best.pt'
    assert files['data/photo.jpg']==(data/'photo.jpg').resolve()
    assert 'assets/train_targets.pt' not in files
    assert 'run.py' not in files
    assert 'delivery.py' in files
    manifest=json.loads(generated['assets/manifest.json'])
    assert manifest['files']['best.pt']==pack.BEST_83125
    assert set(manifest['files'])=={'best.pt','processor/config.json'}


def test_release_rejects_wrong_image_or_checkpoint(tmp_path,monkeypatch):
    args=fixture_files(tmp_path,monkeypatch)
    (args[1]/'photo.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='protocol image'):pack.enrolled_files(*args)
    (args[2]/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='pinned best'):pack.enrolled_files(*args)


def test_release_rejects_wrong_reference(tmp_path,monkeypatch):
    args=fixture_files(tmp_path,monkeypatch)
    path=args[2]/'verification/final_report.json'
    obj=json.loads(path.read_text());obj['metrics']['normal']['hit_at_1']=.83;path.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='verification'):pack.enrolled_files(*args)
