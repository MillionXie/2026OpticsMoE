"""Original image semantics/fresh workload planning without model or CUDA."""
import ast
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run_abo_image_student import trial_plan
from measure_abo_image_student import common_work,BODY_SHA


def test_separate_fresh_bank_then_three_fresh_queries():
    assert trial_plan([1,2,3],None)==[('bank','offline_title_bank',None),('query','ours_repeat01',1),('query','ours_repeat02',2),('query','ours_repeat03',3)]
    assert trial_plan([2,3],Path('existing_bank'))==[('query','ours_repeat02',2),('query','ours_repeat03',3)]


@pytest.mark.parametrize('values',[[],[1,1],[2,1],[4]])
def test_only_missing_unique_rounds(values):
    with pytest.raises(ValueError):trial_plan(values,None)


def test_actual_common_work_is_paired_by_original_sample_not_internal_student_tokens():
    test=[dict(sample_id=str(i),product_id='p') for i in range(200)]
    reference=[dict(index=i,**r,valid_input_tokens=68+(i<128)) for i,r in enumerate(test)]
    assert sum(common_work(reference,test))==13728
    reference[0]['sample_id']='different'
    with pytest.raises(ValueError):common_work(reference,test)


def test_correct_body_original_cpu_nonstable_ranking_and_no_training():
    source=(HERE/'measure_abo_image_student.py').read_text();ast.parse(source)
    assert BODY_SHA.startswith('a2aa9a93') and 'stable=True' not in source
    assert 'original._encode_titles_student' in source and 'original.GroceryRetrievalDataset' in source
    assert 'online_title_bank_forwards=0' in source and 'formal_full2400_reference_hit1=.8058333333333333' in source
    assert 'initialize_student(' not in source and '.backward(' not in source
    assert "counts['bridge.native_merger']!=200" in source


def test_transient_gap_does_not_trigger_immediate_reload():
    source=(HERE/'run_abo_image_student.py').read_text()
    assert "'--idle-checks',type=int,default=12" in source
    assert 'while idle<args.idle_checks:' in source and 'if records:idle=0;' in source
    assert 'not 3<=args.idle_checks<=120' in source
