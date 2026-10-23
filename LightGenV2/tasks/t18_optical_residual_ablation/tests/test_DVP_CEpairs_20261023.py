"""Old camera continuation must keep its own native parent and physical lineage."""
import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]


def test_original285_parent_and_no_camera_batch_mix():
    p=json.loads((HERE/'lab_caltech20_L4_DVP_CEpairs_profile_20261023.json').read_text())
    assert p['parent_correct']==285 and p['target']==340 and p['NLL_tie_tolerance']==1e-4
    assert p['parent_SHA256']=='c172de2221c433c273558a5ce2bfc3a1f0d6d9e5c91a717bdab9c924f2f634d2'
    assert 'caltech20_residual_DVP_46_20261020/' in p['parent_PT']
    assert 'TUCam' not in p['parent_PT'] and 'TUCam' not in p['final_capture']
    assert p['CE_pairs']['replans']==4 and p['CE_pairs']['fit_seconds']==180 and p['CE_pairs']['rows']['fit_seconds']==90


def test_reuses_tested_CE_curvature_but_no_test_gradient():
    text=(HERE/'lab_caltech20_dvp_terminal_reopen_20261021.py').read_text();ast.parse(text)
    assert "ce_pairs=cfg.get('method')=='TRAIN_adaptive_CE_pairs';head_only=head_only or ce_pairs" in text
    assert "pair_repair,row_newton" in text and 'TEST_examples_in_training=0' in text
    assert "fresh['correct']==expected" in text and 'np.max(np.abs(fp-cached_prob))<=3e-6' in text
    assert 'lastw,lastb=latest' in text and 'lastw,lastb=retained()' not in text
    assert 'if best>=340 or (best,bnll)==before:break' in text
