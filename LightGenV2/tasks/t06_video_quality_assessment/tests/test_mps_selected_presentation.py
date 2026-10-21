import matplotlib.pyplot as plt
import numpy as np
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.reports import present_mps_dataset_once as report


@pytest.mark.parametrize('metric',['time16_ms','energy16_j'])
def test_all_six_model_pairs_are_connected_inside_axes_without_jitter(metric):
    points=[]
    for i,model in enumerate(report.NAMES):
        for cap in (100,50):
            points.append(dict(model=model,resource_cap=cap,time16_ms=20*(i+1)+(cap==50),
                energy16_j=2*(i+1)+(cap==50)*.1,srcc=.70+i*.02,accuracy_reference_cap=100))
    fig,ax=plt.subplots()
    try:
        receipt=report.panel(ax,points,metric)
        assert receipt['connected_pairs']==6 and receipt['plotted_points']==13
        assert len(ax.lines)==6 and len(ax.collections)==13
        for i,model in enumerate(report.NAMES):
            open_mark,filled_mark=ax.collections[1+2*i:3+2*i]
            assert np.array_equal(open_mark.get_paths()[0].vertices,filled_mark.get_paths()[0].vertices)
            expected=[p for p in points if p['model']==model]
            assert list(ax.lines[i].get_xdata())==[p[metric] for p in expected]
            assert np.allclose(open_mark.get_offsets()[0],[expected[1][metric],expected[1]['srcc']])
            assert np.allclose(filled_mark.get_offsets()[0],[expected[0][metric],expected[0]['srcc']])
        assert 'batch' not in ax.get_xlabel().lower() and 'N=16' not in ax.get_xlabel()
    finally:
        plt.close(fig)
