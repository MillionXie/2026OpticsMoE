import pytest
from LightGenV2.tasks.t06_video_quality_assessment import rental_followup_audit as a

def test_tied_rank_and_pearson():
    assert a.ranks([3,1,1,4])==[3,1.5,1.5,4]
    a.near(a.correlation([1,2,3],[2,4,6]),1)
    a.near(a.correlation(a.ranks([1,2,3]),a.ranks([3,2,1])),-1)
    with pytest.raises(ValueError):a.near(.8,.7)
