"""Read-only contracts for adopted-body CPU numerical preparation, not GPU results."""
import ast
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from abo_student_graph import BODY_SHA,PROFILE


def test_adopted_profile_and_body_not_old_point88_or_T07():
    from LightGenV2.tasks.t08_abo_image_text_retrieval.text_to_image import BODY_SHA256,PROFILE_SHA256
    import hashlib
    assert BODY_SHA==BODY_SHA256
    assert hashlib.sha256(PROFILE.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PROFILE_SHA256


def test_actual_original_neural_forward_and_no_gpu_in_contract_entry():
    text=Path(__file__).with_name('abo_student_graph.py').read_text()
    tree=ast.parse(text)
    assert 'original._encode_titles_student(' in text and 'original.student_embeddings(' in text
    assert 'torch.equal(expected,actual)' in text
    assert 'torch.cuda.is_initialized()' in text
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in ast.walk(tree))
