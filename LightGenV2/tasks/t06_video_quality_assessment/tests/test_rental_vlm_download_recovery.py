from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_download_recovery as r

@pytest.mark.parametrize("offset,total",[(0,17),(3,17),(0,1000000000),(6723469312,8591720216)])
def test_exact_contiguous_bounded_ranges(offset,total):
    bounds=r.range_plan(offset,total)
    assert bounds[0][0]==offset
    assert len(bounds)<=12
    assert bounds[-1][1]<total
    assert all(end-start+1<=r.CHUNK for start,end in bounds)
    assert all(a[1]+1==b[0] for a,b in zip(bounds,bounds[1:]))

def test_finished_and_invalid_prefix():
    assert r.range_plan(10,10)==[]
    with pytest.raises(ValueError):r.range_plan(11,10)
    with pytest.raises(ValueError):r.range_plan(-1,10)

def test_preserved_prefix_and_complete_official_sha():
    text=Path(r.__file__).read_text()
    assert 'partial.open("ab")' in text
    assert 'prefix_sha=sha(partial)' in text
    assert 'if not check(partial,spec)' in text
    assert 'partial.rename(target)' in text
    assert 'time.sleep(delay)' in text and 'min(delay,60)' not in text
    assert r.DEADLINE<1792185600
