"""Static safety contracts for untimed original student-gallery preparation."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).with_name('prepare_abo_student_gallery.py').read_text(encoding='utf8')
TREE=ast.parse(SOURCE)


def test_parent_never_imports_torch_before_fork():
    parent=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assert not any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in ast.walk(parent))
    assert 'os.fork()' in ast.unparse(parent)
    assert ast.unparse(parent).index('cpu_contract.command.json')<ast.unparse(parent).index('engine.daemon(True)')


def test_cpu_actual_images_outside_first200_and_original_math():
    cpu=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='cpu_contract')
    text=ast.unparse(cpu)
    assert 'for index in (200, 2399)' in text
    assert text.count('original.encode_student_samples(')==2
    assert 'torch.equal(expected, actual)' in text
    assert 'torch.cuda.is_initialized()' in text


def test_no_teacher_cache_or_saved_half_gallery_substitution():
    worker=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    text=ast.unparse(worker)
    assert 'original.encode_student_samples(' in text
    assert 'settings.inference_batch_size != 1' in text
    assert 'vectors.float().numpy()' in text
    assert 'torch.load(' not in text
    assert 'title_or_dummy_inference_calls=0' in text


def test_no_training_or_competing_job_termination():
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('backward','step','train') for n in ast.walk(TREE))
    assert 'os.kill(child, signal.SIGTERM)' in ast.unparse(TREE)
    assert 'if snapshot[\'foreign_gpu_processes\']' in ast.unparse(TREE)


def source_scope(tmp_path):
    for name in ('prepare_abo_student_gallery.py','abo_student_graph.py','real.py'):
        (tmp_path/name).write_text('original source\n')
    scope=dict(Path=Path,REPO=tmp_path,HERE=tmp_path,__file__=str(tmp_path/'prepare_abo_student_gallery.py'),
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),json=json,
        sys=SimpleNamespace(modules={'virtual':SimpleNamespace(__file__='_classes.py'),
            'real':SimpleNamespace(__file__=str(tmp_path/'real.py'))}))
    functions=[n for n in TREE.body if isinstance(n,ast.FunctionDef)
        and n.name in ('imported_sources','generated_module_files','validate_reused_image_contract')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'source_registration_contract','exec'),scope)
    return scope


def test_torch_virtual_relative_source_is_explicit_not_a_git_file(tmp_path):
    scope=source_scope(tmp_path)
    assert set(scope['imported_sources']())=={'prepare_abo_student_gallery.py','abo_student_graph.py','real.py'}
    assert scope['generated_module_files']()==['_classes.py']


def test_missing_absolute_real_source_still_fails(tmp_path):
    scope=source_scope(tmp_path);scope['sys'].modules['missing']=SimpleNamespace(__file__=str(tmp_path/'missing.py'))
    with pytest.raises(FileNotFoundError):scope['imported_sources']()


def test_reuse_only_already_passed_exact_image_contract(tmp_path):
    scope=source_scope(tmp_path);title=tmp_path/'title.json';title.write_text('{}')
    data=dict(status='passed',no_gpu=True,title_contract_sha256=scope['sha'](title),
        identity=dict(body_sha256='cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'),
        checks=[dict(test_index=i,bitwise_equal=True) for i in (200,2399)])
    image=tmp_path/'image.json';image.write_text(json.dumps(data))
    assert scope['validate_reused_image_contract'](image,title)==data
    data['checks'][1]['bitwise_equal']=False;image.write_text(json.dumps(data))
    with pytest.raises(ValueError):scope['validate_reused_image_contract'](image,title)
