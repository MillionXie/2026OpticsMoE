"""Source/config and dispatch contracts; no Torch, dataset, SVG or device execution."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[3]
TASK=ROOT/'LightGenV2/tasks/t04_semantic_interaction'


def test_imported_application_sources_match_pinned_original_bytes():
    identity=json.loads((TASK/'layered_application_import_20261007.json').read_text(encoding='utf8'))
    assert len(identity['files'])==12
    for row in identity['files']:
        payload=(ROOT/row['path']).read_bytes().replace(b'\r\n',b'\n')
        assert len(payload)==row['bytes_lf']
        assert hashlib.sha256(payload).hexdigest()==row['sha256_lf']
        if row['path'].endswith('.py'):
            ast.parse(payload)


def test_selected_application_pt_cannot_be_mispackaged_as_old_standard_head(tmp_path):
    identity=json.loads((TASK/'layered_application_import_20261007.json').read_text(encoding='utf8'))['selected_application']
    assert identity['epoch']==45
    assert identity['checkpoint_sha256']=='03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21'
    assert identity['changed_cell_accuracy_simulation']==.8765
    assert identity['same_checkpoint_remove_optical']==.4845
    assert identity['physical_accuracy_for_this_checkpoint'] is None
    assert 'electronicexp0p50' in identity['architecture']
    run=tmp_path/'run';run.mkdir()
    (run/'resolved_config.json').write_text('{}',encoding='utf8')
    tree=ast.parse((TASK/'build_lab_package.py').read_text(encoding='utf8'))
    build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build')
    namespace={'json':json}
    exec(compile(ast.Module(body=[build],type_ignores=[]),'build_lab_package.build','exec'),namespace)
    torch=ModuleType('torch')
    torch.load=lambda *args,**kwargs: {'architecture':identity['architecture'],'epoch':45}
    safe=ModuleType('safetensors.torch');safe.save_file=lambda *args,**kwargs: None
    with patch.dict(sys.modules,{'torch':torch,'safetensors':ModuleType('safetensors'),'safetensors.torch':safe}):
        with pytest.raises(ValueError,match='exclusively OURS standard-head epoch40'):
            namespace['build'](run,tmp_path/'data',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def merged_config(name, seen=()):
    assert name not in seen
    value=yaml.safe_load((TASK/'configs'/name).read_text(encoding='utf8'))
    parent=value.pop('base_config',None)
    result=merged_config(parent,seen+(name,)) if parent else {}
    def merge(target,source):
        for key,item in source.items():
            if isinstance(item,dict) and isinstance(target.get(key),dict):
                merge(target[key],item)
            else:
                target[key]=item
    merge(result,value)
    return result


def test_layered_config_graph_preserves_layout_and_compresses_only_residual_width():
    reference=merged_config('layered_scene_focus_changed_iou.yaml')
    for name,expansion in [('layered_scene_electronic_exp1.yaml',1.0),('layered_scene_electronic_exp05.yaml',.5)]:
        value=merged_config(name)
        assert value['dataset']==reference['dataset']
        assert value['loss']==reference['loss']
        assert value['training']==reference['training']
        original=dict(reference['model']); compressed=dict(value['model'])
        assert compressed.pop('electronic_expansion')==expansion
        original.pop('electronic_expansion',None)
        assert compressed==original
    assert reference['dataset']['layout_version']=='layered_anchor6_svg_v3'
    assert reference['dataset']['train_samples']==5000
    assert reference['dataset']['test_samples']==1000


@pytest.mark.parametrize('baseline',[False,True])
@pytest.mark.parametrize('layered',[False,True])
def test_actual_data_dispatch_uses_correct_renderer_and_cache(baseline,layered):
    tree=ast.parse((TASK/'run.py').read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ensure_data')
    namespace={'Any':object,'torch':SimpleNamespace(device=object)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'run._ensure_data','exec'),namespace)
    calls=[]
    prefix='LightGenV2.tasks.t04_semantic_interaction'
    fake={}
    for suffix,function,marker in [('embedding_data','prepare_embedding_data','grid'),
                                   ('layered_scene_data','prepare_layered_embedding_data','layered'),
                                   ('qwen_shared','prepare_native_cache','cache')]:
        module=ModuleType(prefix+'.'+suffix)
        def callback(*args,marker=marker):
            calls.append(marker)
            return {'renderer':marker}
        setattr(module,function,callback)
        fake[module.__name__]=module
    namespace['__package__']=prefix
    settings=SimpleNamespace(qwen_shared_baseline=baseline,embedding_only=not baseline,
                             layout_version='layered_anchor6_svg_v3' if layered else 'grid_v2')
    with patch.dict(sys.modules,fake):
        result=namespace['_ensure_data'](settings,None)
    renderer='layered' if layered else 'grid'
    assert result=={'renderer':renderer}
    assert calls==[renderer]+(['cache'] if baseline else [])


def test_profiles_and_gallery_are_explicit_without_changing_train_function():
    identity=json.loads((TASK/'layered_application_import_20261007.json').read_text(encoding='utf8'))
    run=ast.parse((TASK/'run.py').read_text(encoding='utf8'))
    data=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='_ensure_data')
    assert hashlib.sha256(ast.dump(data,include_attributes=False).encode()).hexdigest()==identity['projection_guards']['original_ensure_data_ast_sha256']
    profiles=next(ast.literal_eval(n.value) for n in run.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='PROFILES' for t in n.targets))
    expected=[Path(r['path']).name for r in json.loads((TASK/'layered_application_import_20261007.json').read_text(encoding='utf8'))['files'] if r['path'].endswith('.yaml')]
    assert set(expected).issubset(set(profiles.values()))
    training=ast.parse((TASK/'training.py').read_text(encoding='utf8'))
    train=next(n for n in training.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    assert hashlib.sha256(ast.dump(train,include_attributes=False).encode()).hexdigest()==identity['projection_guards']['unchanged_train_ast_sha256']
    evaluate=next(n for n in training.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_selected')
    dispatch=[n for n in ast.walk(evaluate) if isinstance(n,ast.If)
              and isinstance(n.test,ast.Compare) and 'layered_anchor6_svg_v3' in ast.unparse(n.test)]
    assert len(dispatch)==1
    assert 'save_layered_gallery' in ast.unparse(dispatch[0].body)
    assert 'legacy._save_gallery' in ast.unparse(dispatch[0].orelse)
