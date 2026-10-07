import hashlib
from LightGenV2.tasks.t04_semantic_interaction.verify_layered_assets import inspect


def test_identity_gate_is_read_only_and_rejects_changed_or_missing_assets(tmp_path):
    data = tmp_path/'data'; svg = tmp_path/'svg'
    data.mkdir(); svg.mkdir()
    (data/'train.jsonl').write_bytes(b'{}\n{}\n')
    (svg/'icon.svg').write_bytes(b'<svg/>')
    expected = {'dataset_files': {'train.jsonl': {'sha256': hashlib.sha256(b'{}\n{}\n').hexdigest(), 'records': 2}},
                'svg_files': [{'path': 'icon.svg', 'sha256': hashlib.sha256(b'<svg/>').hexdigest()}]}
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = inspect(data, svg, expected)
    assert not result['errors'] and result['read_only']
    assert not result['models_loaded'] and not result['test_evaluated']
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    (data/'train.jsonl').write_bytes(b'changed')
    assert inspect(data, svg, expected)['errors'][0]['error'] == 'sha256_mismatch'
    missing = tmp_path/'missing'
    assert inspect(missing, svg, expected)['errors'][0]['error'] == 'missing'
    assert not missing.exists()
