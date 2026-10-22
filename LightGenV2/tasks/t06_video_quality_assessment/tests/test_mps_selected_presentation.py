import matplotlib.pyplot as plt
import numpy as np
import pytest
import json
from LightGenV2.tasks.t06_video_quality_assessment.reports import present_mps_dataset_once as report


def test_plain_model_labels_do_not_mutate_source_names_or_measurements():
    names=dict(qwen='Qwen3-VL-2B†',clip='CLIP ViT-B/32')
    assert report.plain_model_names(names)==dict(qwen='Qwen3-VL-2B',clip='CLIP ViT-B/32')
    assert names['qwen']=='Qwen3-VL-2B†'


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


def test_fixed_batch6_refuses_incomplete_audit_before_overwriting_exports(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'REPO',tmp_path)
    prior=tmp_path/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_batch_activity_a100_20261021/resident01'
    prior.mkdir(parents=True)
    (prior/'selected_batch_paper.json').write_text(json.dumps(dict(points=[])))
    (prior/'historical_80gb_published_audit.json').write_text(json.dumps(dict(status='passed')))
    root=prior.parent/'batch6_01';root.mkdir()
    (root/'final_saved_audit.json').write_text(json.dumps(dict(status='passed',audited_points=24,suite_status='running',mps_released=False)))
    with pytest.raises(AssertionError):report.selected_batch_view(root,6)
    assert not (root/'batch6_paper.json').exists()


def test_fixed_batch_plot_keeps_all_three_repeats_and_both_caps(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'REPO',tmp_path);monkeypatch.setattr(report,'OUT',tmp_path)
    for batch in (6,10):
        root=tmp_path/f'batch{batch}_01';root.mkdir()
        groups={model:dict(caps={str(cap):dict(batch=batch,time16_ms=100,energy16_j=40,
            time_repeats_ms=[99,100,101],energy_repeats_j=[39,40,41]) for cap in (100,50)}) for model in ('clip','yolo')}
        (root/'independent_cost_check.json').write_text(json.dumps(dict(status='passed',models=groups,independent_time_and_overlap_energy_replay=True)))
    report.fixed_batch_comparison(tmp_path/'batch6_01')
    receipt=json.loads((tmp_path/'batch6_01/batch6_batch10_figure_receipt.json').read_text())
    assert len(receipt['rows'])==8 and len(receipt['files'])==4
    assert all(len(r['time_repeats_ms'])==len(r['energy_repeats_j'])==3 for r in receipt['rows'])
