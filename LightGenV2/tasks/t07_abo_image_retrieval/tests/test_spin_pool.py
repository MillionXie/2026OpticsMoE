import csv
import argparse
import hashlib
import io
import json

import pytest
from PIL import Image

from LightGenV2.tasks.t07_abo_image_retrieval.standalone import prepare_spin_abo as spin
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.enrolled_regularization import load_external_pool, spin_target_identity
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256


def test_documented_target_and_metadata_sha_are_valid_64hex():
    digest = 'f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde'
    assert spin.sha256_argument(digest) == digest
    assert spin.sha256_argument(digest.upper()) == digest
    assert spin.sha256_argument(spin.METADATA_SHA) == spin.METADATA_SHA
    for bad in (digest + 'e', digest[:-1], 'z' * 64, digest + '\n'):
        with pytest.raises(argparse.ArgumentTypeError): spin.sha256_argument(bad)


def test_uniform_views_use_actual_sequence_not_random_image_duplicates():
    rows = [dict(azimuth=str(i)) for i in reversed(range(72))]
    assert [int(r['azimuth']) for r in spin.uniform_views(rows, 12)] == list(range(0, 72, 6))
    assert len(spin.uniform_views(rows[::3], 12)) == 12
    for bad in ([dict(azimuth=0)] * 12, [dict(azimuth=72)], rows[:2]):
        with pytest.raises(ValueError): spin.uniform_views(bad, 12)


def test_candidates_exclude_entire_target_spin_and_aliases_and_are_deterministic():
    def listing(pid, sid):
        return dict(item_id=pid, spin_id=sid, product_type=[dict(value='CHAIR')])
    listings = [listing('target', 'a'), listing('different_sku_same_target_spin', 'a'),
                listing('alias1', 'b'), listing('alias2', 'b'), listing('safe1', 'c'), listing('safe2', 'd')]
    spins = {sid: [dict(azimuth=i) for i in range(12)] for sid in 'abcd'}
    target = [dict(product_id='target', spin_id='a')]
    a, stats = spin.candidates(listings, spins, target, 4)
    b, _ = spin.candidates(reversed(listings), spins, target, 4)
    assert a == b and {v[0] for v in a['CHAIR']} == {'safe1', 'safe2'}
    assert stats['target_sku_or_spin'] == 2 and stats['shared_spin_sku_alias'] == 2


def test_download_budget_and_atomic_cache_do_not_leave_partial_files(tmp_path, monkeypatch):
    data = b'bounded official test data'
    def opened(*args, **kwargs):
        f = io.BytesIO(data); f.headers = {'Content-Length': str(len(data))}; return f
    monkeypatch.setattr(spin.urllib.request, 'urlopen', opened)
    url = spin.BASE + 'test'; dest = tmp_path / 'test'
    with pytest.raises(spin.BudgetExceeded):
        spin.fetch_file(url, dest, spin.DownloadBudget(2), 100)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        spin.fetch_file(url, dest, spin.DownloadBudget(100), 100, 'wrong')
    assert not list(tmp_path.iterdir())
    budget = spin.DownloadBudget(100)
    digest = hashlib.sha256(data).hexdigest()
    assert spin.fetch_file(url, dest, budget, 100, digest).read_bytes() == data
    assert budget.used == len(data)
    spin.fetch_file(url, dest, budget, 100, digest)
    assert budget.used == len(data) and len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError): spin.fetch_file('https://other.invalid/file', dest, budget, 100)


def fixture(tmp_path):
    rows = []
    for i in range(4):
        path = tmp_path / f'{i}.png'; Image.new('RGB', (16, 16), (i * 50, 40, 100)).save(path)
        rows.append(dict(sample_id=str(i), product_id=f'ext{i // 2}', category_id=0,
                         spin_id=f'spin{i // 2}', azimuth=i % 2, image_path=path.name, image_sha256=sha256(path)))
    protected = [dict(product_id='target', spin_id='protected', image_sha256='no', split='train')]
    protocol = dict(protocol='abo200_enrolled_sku_hash8train4query_v1', parent_manifest_sha256='parent', rows=protected)
    groups = dict(train=protected, query=[])
    report = dict(status='ready', pool_kind='abo_spin_pretrain_v1', target_manifest_sha256='parent',
                  enrolled_rows_sha256=spin_target_identity(protocol), selected_images=4, selected_products=2, views_per_product=2)
    return rows, report, protocol, groups


@pytest.mark.parametrize('bad', [None, 'target_spin', 'alias', 'partition', 'duplicate_angle', 'incomplete', 'two_spins'])
def test_loader_checks_target_partition_spin_identity_and_view_counts(tmp_path, bad):
    rows, report, protocol, groups = fixture(tmp_path)
    if bad == 'target_spin': rows[0]['spin_id'] = 'protected'
    if bad == 'alias': rows[2]['spin_id'] = 'spin0'
    if bad == 'partition': protocol['rows'][0]['split'] = 'query'
    if bad == 'duplicate_angle': rows[1]['azimuth'] = 0
    if bad == 'incomplete': report['status'] = 'failed'
    if bad == 'two_spins': rows[1]['spin_id'] = 'another'
    path = tmp_path / 'manifest.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    digest = sha256(path); report['manifest_sha256'] = digest
    (tmp_path / 'report.json').write_text(json.dumps(report))
    if bad:
        with pytest.raises(ValueError): load_external_pool(tmp_path, tmp_path, protocol, groups, digest)
    else:
        fit, audit = load_external_pool(tmp_path, tmp_path, protocol, groups, digest)
        assert len(fit['train']) == 4 and audit['target_spin_overlap'] == 0
