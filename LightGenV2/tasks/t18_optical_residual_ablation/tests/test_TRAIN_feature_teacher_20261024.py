import ast
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_feature_teacher_20261024.py'


def test_original_teacher_only_TRAIN_and_no_mutating_training():
    s=P.read_text();ast.parse(s)
    assert "m.padded(data,'train',begin)" in s and "data['test']" not in s and "model.measured={}" in s
    assert 'torch.optim' not in s and '.backward(' not in s and "model.cpu().eval().requires_grad_(False)" in s
    assert 'source_PT_SHA256=m.sha(pt)' in s and "role='train'" in s and 'TEST_used=False' in s
    assert 'error<=3e-6' in s and 'original_state_bitwise_unchanged=True' in s
