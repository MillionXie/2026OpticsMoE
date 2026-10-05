import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

source = Path(__file__).resolve().parents[3] / 'LightGenV2/tasks/t01_object_retrieval/asset_binding.py'
sys.path.insert(0, str(source.parents[3]))
spec = importlib.util.spec_from_file_location('t01_asset_binding', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    files = {}
    for name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json',
                 'preprocessor_config.json','chat_template.jinja'):
        data = ('fixture '+name).encode()
        (snapshot/name).write_bytes(data)
        files[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    manifest=tmp_path/'identity.json'
    manifest.write_text(json.dumps({'frozen_frontend':{'model_id':'test/model','files':files}}))
    return snapshot,manifest


def test_default_is_noop():
    settings=SimpleNamespace(model_id='original',teacher_cache_path='old')
    before=vars(settings).copy()
    assert module.bind_frontend(settings,SimpleNamespace()) is None
    assert vars(settings)==before


def test_binding_verified_offline_and_fresh_cache(tmp_path):
    snapshot,manifest=fixture(tmp_path)
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import Settings
    settings=Settings.__new__(Settings)
    settings.model_id='test/model'
    output=tmp_path/'fresh'
    result=module.bind_frontend(settings,SimpleNamespace(frontend_snapshot=snapshot,frontend_identity=manifest,run_dir=output))
    assert result['files_verified']==6 and result['historical_run_revision_proven'] is False
    assert settings.local_files_only is True and settings.model_id==str(snapshot)
    assert settings.teacher_cache_path==output/'teacher_cache/teacher_embeddings.pt'
    assert not output.exists()


def test_existing_run_refused_before_settings_mutation(tmp_path):
    snapshot,manifest=fixture(tmp_path)
    settings=SimpleNamespace(model_id='old')
    with pytest.raises(ValueError,match='nonempty'):
        module.bind_frontend(settings,SimpleNamespace(frontend_snapshot=snapshot,frontend_identity=manifest,run_dir=snapshot))
    assert vars(settings)=={'model_id':'old'}


def test_content_mismatch_refused(tmp_path):
    snapshot,manifest=fixture(tmp_path)
    (snapshot/'config.json').write_bytes(b'x'*(snapshot/'config.json').stat().st_size)
    with pytest.raises(ValueError,match='SHA mismatch'):
        module.verify_snapshot(snapshot,manifest)


def test_partial_binding_refused():
    with pytest.raises(ValueError,match='requires'):
        module.bind_frontend(SimpleNamespace(),SimpleNamespace(frontend_snapshot='x',run_dir='y'))


def test_different_backbone_cannot_replace_profile(tmp_path):
    snapshot,manifest=fixture(tmp_path)
    settings=SimpleNamespace(model_id='original/model')
    with pytest.raises(ValueError,match='differs'):
        module.bind_frontend(settings,SimpleNamespace(frontend_snapshot=snapshot,frontend_identity=manifest,run_dir=tmp_path/'out'))
    assert vars(settings)=={'model_id':'original/model'}


def test_unsafe_manifest_path_refused(tmp_path):
    snapshot,manifest=fixture(tmp_path)
    value=json.loads(manifest.read_text())
    value['frozen_frontend']['files']['../escape']={'bytes':0,'sha256':'0'*64}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Unsafe'):
        module.verify_snapshot(snapshot,manifest)
