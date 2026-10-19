"""Layout-only fixture; these temporary figures are never experiment evidence."""
import json
import shutil
from LightGenV2.tasks.t06_video_quality_assessment.reports import precision_comparison_20261019 as m


def test_four_panel_layout_at_word_width(tmp_path,monkeypatch):
    old=m.read(m.OLD)['groups']['busy50_near_saturation']
    fixture=[]
    for row in old:
        if row.get('plotted',True):fixture.append(dict(row))
        else:
            high=next(x for x in old if x['baseline']==row['baseline'] and x['condition']=='Near-saturation reference')
            fixture.append(dict(high,condition='Approximately 50%',utilization_percent=50,plotted=True))
    original=m.OUT;out=tmp_path/'out';out.mkdir()
    names=['README.md','LGVQ_busy50_near_saturation_plot_data.xlsx','LGVQ_busy50_near_saturation_plot_data.csv','busy50_near_saturation_pair.png']
    names += [f'busy50_near_saturation_{kind}.{ext}' for kind in ('latency','energy') for ext in ('png','svg','pdf')]
    for name in names:shutil.copy2(original/name,out/name)
    run=tmp_path/'layout_only';(run/'presentation').mkdir(parents=True)
    (run/'presentation/precision_comparison.json').write_text(json.dumps(dict(groups={'native':fixture,'fp32':fixture},missing=[],fixture_not_measured_FP32=True)))
    monkeypatch.setattr(m,'OUT',out);monkeypatch.setattr(m,'RENTAL',run)
    m.render()
    result=m.read(run/'presentation/figure_geometry.json')
    assert result['status']=='passed' and len(result['panels'])==4
    assert all(p['labels']==13 and p['overlaps']==0 for p in result['panels'])
