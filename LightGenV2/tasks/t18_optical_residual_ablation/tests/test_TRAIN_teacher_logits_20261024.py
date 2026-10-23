import sys,json,ast
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from lab_caltech20_TRAIN_teacher_logits_20261024 import targets,selftest


def test_label_safe_centered_logit_targets():
    assert selftest()['passed']
    y=np.arange(20);z=np.eye(20)*8+np.arange(20)[:,None]*100
    a,g=targets(y,z);b,_=targets(y,np.eye(20)*8)
    assert g.all() and np.allclose(a,b) and np.allclose(np.mean(a*a,axis=1),1)


def test_no_TEST_fit_and_predeclared_TRAIN_support():
    s=(HERE/'lab_caltech20_TRAIN_teacher_logits_20261024.py').read_text();ast.parse(s)
    assert 'test_labels' not in s and 'import torch' not in s and 'correct>=873' in s and 'len(np.unique(families))==1508' in s
    p=json.loads((HERE/'lab_caltech20_actual300_LOGIT285_profile_20261024.json').read_text())
    assert p['expected_parent_correct']==285 and p['TRAIN_teacher_target_kind']=='normalized_logits'
