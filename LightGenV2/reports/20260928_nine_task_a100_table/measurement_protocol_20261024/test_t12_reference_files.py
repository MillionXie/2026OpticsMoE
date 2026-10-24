import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from t12_reference_files import validate_records,CONTRACT


def records():
    return [dict(index=i,sample_id=f'original:{i}',image=f'reference_{i:03d}.png',
        text=f'instruction_{i:03d}.txt',image_size=[256,256],reference_bitwise_equal=True,
        online_inputs_contain_target=False,png_sha256='a'*64,utf8_sha256='b'*64) for i in range(200)]


def test_fixed_first200_original_input_contract():
    validate_records(records())
    assert CONTRACT=='original_T12_first200_reference_RGB_and_UTF8_v1'


@pytest.mark.parametrize('change',('target','duplicate','wrong_pixels','permutation','escape'))
def test_reject_wrong_reference_input_contract(change):
    rows=records()
    if change=='target':rows[0]['online_inputs_contain_target']=True
    if change=='duplicate':rows[1]['sample_id']=rows[0]['sample_id']
    if change=='wrong_pixels':rows[0]['reference_bitwise_equal']=False
    if change=='permutation':rows[0],rows[1]=rows[1],rows[0]
    if change=='escape':rows[0]['image']='../target.png'
    with pytest.raises(ValueError):validate_records(rows)
