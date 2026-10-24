"""Static safety contracts for untimed original student-gallery preparation."""
import ast
from pathlib import Path

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
