import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import lab_fp32_recheck as m


def test_no_autocast_or_storage_downcast():
    source=Path(m.__file__).read_text()
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('autocast','half','bfloat16') for n in calls)
    assert 'allow_tf32=False' in source and "set_float32_matmul_precision('highest')" in source


def test_canonical_and_anchor_protection():
    source=Path(m.__file__).read_text()
    assert "('qwen',3)" in source and "('qwen',1)" not in source
    assert 'no finalized B1 edits' in source
    assert 'test_override' in Path(m.shared.__file__).read_text()


def test_optional_rental_qwen_preserves_anchor_and_boundary():
    source=Path(m.__file__).with_name('rental_qwen_fp32.py').read_text()
    tree=ast.parse(source)
    assert 'args.batch=3' in source and 'original_best_checkpoint.pt' in source
    assert 'args.remapped_test_rows' in source
    assert 'shared.evaluate' in source and 'lab.runtime' in source
    assert 'allow_tf32=False' in source and "set_float32_matmul_precision('highest')" in source
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('half','bfloat16','autocast') for n in ast.walk(tree))
