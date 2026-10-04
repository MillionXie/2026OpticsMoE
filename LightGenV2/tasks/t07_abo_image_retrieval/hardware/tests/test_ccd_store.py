"""Synthetic CCD storage contracts only; no SDK/model/real query evaluation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location('ccd_store', Path(__file__).resolve().parents[1] / 'ccd_store.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def session(tmp_path):
    ids = ['0' * 16, '1' * 16]
    phase = tmp_path / 'phase'
    phase.mkdir()
    hashes = {}
    for stage in module.STAGES:
        path = phase / (stage + '.bmp')
        Image.fromarray(np.zeros((1200, 1920), np.uint8)).save(path)
        hashes[stage] = module.sha(path)
    contract = dict(checkpoint_sha256=module.CHECKPOINT, phase_sha256=hashes,
                    encoding=module.ENCODING, exposure_us=400, gain='Gain_X4', wait_ms=240,
                    geometry=[[586,147],[1379,159],[1369,946],[573,932]],
                    stage_orientation={stage: ['hv_inverse','flip_v'] for stage in module.STAGES}, ids=ids)
    save_json(tmp_path / 'contract.json', contract)
    for stage in module.STAGES:
        folder = tmp_path / 'ccd' / stage
        folder.mkdir(parents=True)
        for sid in ids:
            array = np.full((478,478), 32, np.uint8)
            Image.fromarray(array).save(folder / (sid + '.png'))
            row = dict(stage=stage, sample_id=sid, phase_sha256=hashes[stage], amplitude_sha256='a'*64,
                       exposure=dict(exposure_us=400, gain='Gain_X4'), wait_ms=240,
                       canonical_orientation='flip_v', no_photometric_normalization=True,
                       mean=32., p99=32., maximum=32, saturation_fraction=0.)
            save_json(folder / (sid + '.json'), row)
    return tmp_path, contract


def store(session):
    root, _ = session
    return module.CCDStore(root, contract_sha256=module.sha(root / 'contract.json'))


def test_complete_and_read_only(session):
    root, contract = session
    before = {str(path):module.sha(path) for path in root.rglob('*') if path.is_file()}
    reader = store(session)
    report = reader.audit()
    assert report['complete'] and report['images_per_layer'] == 2
    assert all(row['valid_pairs'] == 2 for row in report['stages'].values())
    assert reader.read('vision_router', contract['ids']).shape == (2,478,478)
    assert before == {str(path):module.sha(path) for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('field,value', [('gain','Gain_X1'),('wait_ms',241),('checkpoint_sha256','b'*64),
                                         ('encoding','peak_scaling'),('ids',['../escape'])])
def test_wrong_contract_rejected(session, field, value):
    root, contract = session
    contract[field] = value
    save_json(root / 'contract.json', contract)
    with pytest.raises(ValueError):
        store(session)


def test_pinned_hash_and_phase_rejected(session):
    root, _ = session
    with pytest.raises(ValueError):
        module.CCDStore(root, contract_sha256='f'*64)
    (root / 'phase/vision_router.bmp').write_bytes(b'changed')
    with pytest.raises(ValueError):
        store(session)


@pytest.mark.parametrize('field,value', [('stage','language_global'),('sample_id','2'*16),
                                         ('phase_sha256','b'*64),('wait_ms',241),
                                         ('canonical_orientation','identity'),('no_photometric_normalization',False),
                                         ('amplitude_sha256','unknown'),('mean',31)])
def test_receipt_mismatch_rejected(session, field, value):
    root, contract = session
    path = root / 'ccd/vision_router' / (contract['ids'][0] + '.json')
    row = json.loads(path.read_text())
    row[field] = value
    save_json(path, row)
    with pytest.raises(ValueError):
        store(session).audit()


def test_receipt_gain_rejected(session):
    root, contract = session
    path = root / 'ccd/vision_router' / (contract['ids'][0] + '.json')
    row = json.loads(path.read_text()); row['exposure']['gain'] = 'Gain_X1'
    save_json(path, row)
    with pytest.raises(ValueError):
        store(session).audit()


def test_partial_and_missing_are_distinct(session):
    root, contract = session
    sid = contract['ids'][0]
    image = root / 'ccd/vision_router' / (sid + '.png')
    receipt = image.with_suffix('.json')
    image.unlink()
    with pytest.raises(ValueError):
        store(session).audit()
    receipt.unlink()
    report = store(session).audit()
    assert not report['complete'] and report['stages']['vision_router']['missing_pairs'] == 1
    with pytest.raises(FileNotFoundError):
        store(session).read('vision_router', [sid])


@pytest.mark.parametrize('gray', [10,255])
def test_dark_or_saturated_rejected(session, gray):
    root, contract = session
    image = root / 'ccd/vision_router' / (contract['ids'][0] + '.png')
    Image.fromarray(np.full((478,478), gray, np.uint8)).save(image)
    row = json.loads(image.with_suffix('.json').read_text())
    row.update(mean=float(gray), p99=float(gray), maximum=gray, saturation_fraction=float(gray==255))
    save_json(image.with_suffix('.json'), row)
    with pytest.raises(ValueError):
        store(session).audit()


def test_unknown_or_duplicate_ids_rejected(session):
    reader = store(session)
    for stage, ids in [('other',['0'*16]), ('vision_router',['2'*16]),
                       ('vision_router',['0'*16,'0'*16]), ('vision_router',[])]:
        with pytest.raises(ValueError):
            reader.read(stage, ids)
