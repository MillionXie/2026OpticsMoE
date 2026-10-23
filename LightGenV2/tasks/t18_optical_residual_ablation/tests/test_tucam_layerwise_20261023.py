"""Static contracts; numerical gradient contracts run on lab CPU, no SDK."""
import ast,json,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_layerwise_20261023.py'

def test_profile_and_integer_gates():
    h=runpy.run_path(str(ENTRY));c=json.loads((HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json').read_text());h['validate_config'](c)
    assert h['gate_target'](346,8)==314 and h['gate_target'](346,1.5)==340
    assert c['initial_SHA256']=='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'
    assert list(c['hardware_by_stage'])==['router','0','1','2','3']

def test_capture_fit_review_then_next_layer():
    s=ENTRY.read_text();tree=ast.parse(s);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    stages=next(n for n in f.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='STAGES');code=ast.get_source_segment(s,stages)
    assert code.index('fields=acquire(')<code.index('fit_suffix(')<code.index("'endpoint_report.json'")
    assert 'interstage_fit=True' in code and "if not endpoint['target_met']:" in code
    assert 'next_layer_not_started=True' in code and 'automatic_retry=False' in code

def test_no_TEST_training_no_new_graph_or_exposure_change():
    s=ENTRY.read_text();tree=ast.parse(s);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_suffix');code=ast.get_source_segment(s,f)
    assert "loop.put_fields(model,data,fields,'train',ix,'cuda')" in code
    assert "F.cross_entropy(held['logits'],labels" in code
    assert 'keys=m.allowed_keys(model,prefix)' in code
    assert 'loop.validate_state(m,native,original,keys)' in code
    assert "for role in ('train','test'):" in code and "m.model_for(pt,m.sha(pt),4,reference)" in code
    assert 'transforms.remove();hook.remove()' in code
    assert 'set_exposure(' not in s and 'physical_adjoint(' not in s and 'Popen(' not in s

def test_windows_evidence_paths_remain_bounded():
    root='E:/code/guest/2026OpticsMoE/LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_20261023'
    assert len(root+'/s_router/capture/train/000000_input.png')<240
    assert len(root+'/s_router/f0/best_train_predictions.npz')<240

def test_metadata_default_preserves_previous_fixed_capture():
    code=(HERE/'lab_caltech20_tucam_capture_20261023.py').read_text();tree=ast.parse(code)
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='acquire')
    assert f.args.args[-1].arg=='interstage_fit' and f.args.defaults[-1].value is False
    assert 'interstage_fit=interstage_fit' in code
