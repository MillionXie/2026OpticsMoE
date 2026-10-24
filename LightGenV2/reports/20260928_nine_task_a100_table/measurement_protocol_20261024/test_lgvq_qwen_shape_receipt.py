from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from lgvq_qwen_shape_receipt import matched


def fixture():
    return dict(calls=600,MAC_OP600=60,repeat_results=[dict(MAC200=10,records=[dict(MAC=10)]) for _ in range(3)],
        cost_references=[dict(seconds200=1,joules200=2,report_sha256=str(i)) for i in range(3)])


def test_exact_original_numerator_and_denominator():
    result=matched(fixture(),3,6,['0','1','2'])
    assert result['TOPS']==20e-12 and result['TOPS_per_W']==10e-12


@pytest.mark.parametrize('seconds,energy,shas',[(2,6,['0','1','2']),(3,5,['0','1','2']),(3,6,['2','1','0'])])
def test_no_different_cost_or_trial_selection(seconds,energy,shas):
    with pytest.raises(ValueError):matched(fixture(),seconds,energy,shas)
