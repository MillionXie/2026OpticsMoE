"""No-hardware contracts for fixed-weight fresh L4 and concurrent CPU fitting."""
import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]


def test_frozen_original_identity_and_sample_budget():
    c=json.loads((HERE/'lab_caltech20_tucam_fixed_L4_profile_20261023.json').read_text())
    assert c['L4_initial_SHA256']=='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'
    assert c['L4_simulation_correct']==346 and c['L4_target']==340 and c['L6_target']==350
    assert c['interstage_fit'] is False and c['formal_L4_frames']==5*(1600+411)
    assert c['VALID']==0 and c['TRAIN_each_class']==80 and len(c['recipes'])==2


def test_no_interstage_fit_and_no_active_L6_mutation():
    code=(HERE/'lab_caltech20_tucam_fixed_L4_20261023.py').read_text();tree=ast.parse(code)
    capture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_L4')
    # The only optimize call is outside the stage loop.
    stage_loop=next(n for n in capture.body if isinstance(n,ast.For))
    assert 'optimize(' not in ast.get_source_segment(code,stage_loop)
    assert "'waiting_existing_L6_capture_audit'" in code and "spawn('capture')" in code and "spawn('optimize6')" in code
    assert 'model.cuda(' not in code and 'camera.reset(' not in code
    assert "all_parameters_bitwise_frozen=True" in code


def test_generic_capture_depth_and_raw_crop_audit():
    code=(HERE/'lab_caltech20_tucam_capture_20261023.py').read_text();ast.parse(code)
    assert "preceding=[] if stage=='router'" in code and 'depth=model.depth' in code
    assert code.count('model.measured={}')>=2 and 'native_crop_recomputed=True' in code
    assert "r['p99']<15 or r['saturation_fraction']>.15" in code
    assert 'from lab_caltech20_tucam_session_20261022 import Bench,logical' in code
    assert "crop=logical(raw,warp,hw['sensor_full_scale'],hw['orientation'])" in code


def test_only_original_terminal_parameters_and_native_review():
    code=(HERE/'lab_caltech20_tucam_terminal_20261023.py').read_text();ast.parse(code)
    assert 'k=model.depth-1' in code and "f'electronic.{k}.local.weight'" in code
    assert 'sum(p.numel() for p in params)==5150' in code
    assert "loop.original_replay(native,data,fields,role)" in code
    assert 'np.max(np.abs(cp-prob))<=3e-6' in code
    assert "gap>cfg['overfit_gap_threshold']" in code
    assert 'if n<8:' in code and "bank[role]=torch.cat(values)[:len(data[role][1])]" in code
    assert 'new_inference_layers=0' in code and 'TEST_gradient=False' in code
