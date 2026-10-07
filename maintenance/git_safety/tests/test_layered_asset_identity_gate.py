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


def test_binding_changes_only_explicit_paths_after_all_guards(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from LightGenV2.tasks.t04_semantic_interaction import verify_layered_assets as gate
    data = tmp_path / 'dataset/openmoji_layered_anchor6_svg_v3'
    data.mkdir(parents=True)
    expected = {'dataset_files': {'train.jsonl': {'sha256': 'trainsha', 'records': 5000},
                                 'test.jsonl': {'sha256': 'testsha', 'records': 1000}}}
    identity = tmp_path/'identity.json'; identity.write_text(json.dumps(expected))
    monkeypatch.setattr(gate, 'IDENTITY', identity)
    monkeypatch.setattr(gate, 'inspect', lambda *a: {'errors': []})
    summary = {'type': 'openmoji_layered_anchor6_proportional_svg_v3', 'seed': 73,
               'train': {'sha256': 'trainsha', 'samples': 5000},
               'test': {'sha256': 'testsha', 'samples': 1000}}
    (data/'dataset_summary.json').write_text(json.dumps(summary))
    settings = SimpleNamespace(layout_version='layered_anchor6_svg_v3', embedding_only=True,
                               qwen_shared_baseline=False, seed=73,
                               prompt_cache_path=tmp_path/'original/token_embeddings_v1.pt',
                               data_dir=tmp_path/'original', svg_asset_dir=tmp_path/'original-svg',
                               electronic_expansion=.5, fusion_alpha_minimum=.4001)
    assert gate.bind_existing(settings, tmp_path) == summary
    assert settings.data_dir == data
    assert settings.prompt_cache_path == data/'token_embeddings_v1.pt'
    assert settings.electronic_expansion == .5 and settings.fusion_alpha_minimum == .4001
    assert not settings.prompt_cache_path.exists()  # mock identity validation did not generate it


def test_failed_binding_keeps_settings_and_assets_untouched(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    from LightGenV2.tasks.t04_semantic_interaction import verify_layered_assets as gate
    settings = SimpleNamespace(layout_version='layered_anchor6_svg_v3', embedding_only=True,
                               qwen_shared_baseline=False, prompt_cache_path=tmp_path/'token_embeddings_v1.pt')
    before = vars(settings).copy()
    monkeypatch.setattr(gate, 'inspect', lambda *a: {'errors': [{'error': 'missing'}]})
    with pytest.raises(ValueError, match='identity check'):
        gate.bind_existing(settings, tmp_path/'absent')
    assert vars(settings) == before and not (tmp_path/'absent').exists()
