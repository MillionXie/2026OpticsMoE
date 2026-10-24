"""No-model exact work/denominator and zero-padding receipt checks."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rental_shape_receipt import matched


def fixture():
    record=dict(MAC=11,MAC_terms=dict(projection=9,padded_AV=2),
        logical_unpadded_MAC=9,FA2_zero_padded_value_dense_MAC=2)
    return dict(calls=600,MAC_OP600=13200,FA2_zero_padding_MAC600=1200,
        repeat_results=[dict(repeat=i,MAC200=2200,records=[deepcopy(record) for _ in range(200)]) for i in (1,2,3)],
        cost_references=[dict(seconds200=i,joules200=i*10,report_sha256=str(i)) for i in (1,2,3)])


def test_exact_triple_and_dense_padding_not_hidden():
    v=matched(fixture(),6,60,['1','2','3'])
    assert v['OP600']==13200 and v['logical_unpadded_OP600']==10800
    assert v['zero_padded_value_dense_OP600']==2400
    assert v['TOPS']==pytest.approx(13200/6e12) and v['TOPS_per_W']==pytest.approx(13200/60e12)


@pytest.mark.parametrize('change', ['report','time','work','padding'])
def test_reject_mismatched_original_receipt(change):
    g=fixture();seconds=6;shas=['1','2','3']
    if change=='report':shas[2]='different'
    elif change=='time':seconds+=.1
    elif change=='work':g['repeat_results'][1]['records'][0]['MAC']+=1
    else:g['FA2_zero_padding_MAC600']+=1
    with pytest.raises(ValueError):matched(g,seconds,60,shas)
