"""No inference rerun: pin original call path, fullTEST accuracy and fresh trials."""
import ast
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import measure_openmoji_qwen as measure


def test_cost_loop_uses_CPU_verified_native_original_not_cache_or_target():
    tree=ast.parse((HERE/'measure_openmoji_qwen.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(index, row)'
              and 'native_and_head(' in ast.unparse(n))
    source=ast.unparse(loop)
    assert 'records[:200]' in source and 'native_and_head(runtime, row, device)' in source
    assert 'target_grid' not in source and 'time.sleep' not in source and 'compose_prediction(' in source
    assert 'torch.cuda.synchronize()' in source and 'cpu_result_return=returned' in source


def test_accuracy_source_fullTEST_and_metrics_only_derived():
    source=(HERE/'measure_openmoji_qwen.py').read_text()
    assert "formal_full1000_reference_metrics=" in source
    assert 'measured_subset_diagnostics_not_formal_accuracy=True' in source
    assert 'interval_energy([(a,b)],telemetry,338.2)' in source
    assert 'operation_ledger' not in source and 'TOPS' not in source


def test_auditor_no_model_or_torch_load_and_full_window_energy():
    source=(HERE/'audit_openmoji_qwen.py').read_text()
    assert 'import torch' not in source and 'torch.load' not in source
    assert 'observed_energy([(a,b)]' in source and '338.2*(b-a)+board' in source
    assert "formal_full1000_reference_metrics" in source and "native_bf16_edit" in source


def test_reject_missing_actual_CPU_contract(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"status":"not_executed"}')
    with pytest.raises(ValueError):measure.validate_contract(p)


def test_controller_fresh_process_reuses_proof_no_CPU_model_or_qualified_rerun():
    source=(HERE/'run_openmoji_qwen.py').read_text()
    assert 'import torch' not in source and 'subprocess.Popen(' in source
    assert 'for repeat in args.repeats' in source and 'sorted(set(args.repeats))' in source
    assert "name=f'qwen_repeat{repeat:02d}'" in source
    assert 'engine.daemon(False)' in source and "measure_openmoji_qwen.py" in source
