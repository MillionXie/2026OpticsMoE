import ast
from pathlib import Path

ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_phase_native_fit_20261024.py'


def test_fixed_photometry_native_response_TRAIN_only():
    s=ENTRY.read_text();ast.parse(s)
    assert 'phase_pupil_parameters=25' in s and 'range(1,21)' in s and 'time.monotonic()-fitstart>=300' in s
    assert 'photometric_map_and_sigma_frozen=True' in s and 'torch.optim.Adam([knots,scale,pupil]' in s
    assert 'rows=[fit[i] for i in order[begin:begin+4]]' in s
    assert "final[tag]['native_optical_PCC']>=baseline[tag]['native_optical_PCC']+.03" in s
    for bad in ("data['test']",'Bench(','capture.acquire(','torch.save(','saved_model(','grid_sample'):assert bad not in s
    assert 'LUT_changed=False,classification_accuracy_updated=False' in s


def test_phase_baseline_reproduces_admitted_response_and_not_prior_rejected_fit():
    s=ENTRY.read_text()
    assert "abs(baseline['held_nominal_before']['native_optical_PCC']-r['fitted']['held_nominal_before']['native_optical_PCC'])<1e-4" in s
    assert 'r[\'native_response_admitted\']' in s and 'knots=torch.nn.Parameter(torch.full((16,)' in s
    assert 'pupil=torch.nn.Parameter(torch.zeros(8))' in s and "'spatial_fit/report.json'" not in s
