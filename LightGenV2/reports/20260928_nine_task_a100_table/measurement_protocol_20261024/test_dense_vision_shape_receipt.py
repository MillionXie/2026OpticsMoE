from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_vision_shape_receipt import matched


def test_match_all_three_original_full_vision_costs():
    group=dict(calls=600,MAC_OP600=1200,
        repeat_results=[dict(repeat=i,MAC200=200,records=[dict(MAC=1,MAC_terms=dict(x=1))]*200) for i in (1,2,3)],
        cost_references=[dict(seconds200=i,joules200=i*10) for i in (1,2,3)],
        all_native24_vision_blocks_and_four_mergers=True,language_or_vocabulary_projection_executed=False)
    assert matched(group,6,60)['TOPS']==pytest.approx(1200/6e12)
    with pytest.raises(ValueError):matched(group,6.1,60)
    group['language_or_vocabulary_projection_executed']=True
    with pytest.raises(ValueError):matched(group,6,60)
