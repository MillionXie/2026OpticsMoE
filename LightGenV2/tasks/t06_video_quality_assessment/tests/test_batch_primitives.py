"""CPU/no-Torch contracts for bounded operator causal diagnostics."""
import ast
from pathlib import Path

from LightGenV2.tasks.t06_video_quality_assessment import diagnose_batch_primitives as primitives


def test_fixed_scope_and_no_training_or_forward_patch():
    assert primitives.BATCHES == (1, 2, 4)
    assert primitives.REFERENCE_LIMIT == 2 * 1024**3
    source = Path(primitives.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {"backward", "train", "step", "sleep"} & calls
    assert not any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                   and n.attr == "forward" for n in ast.walk(tree))
    assert "exist_ok=False" in source and "Instrumentation changed native predictions" in source


def test_real_attention_primitives_and_explicit_numerical_control():
    assert {"aten.mm.default", "aten.bmm.default", "aten._softmax.default",
            "aten.linear.default", "aten.matmul.default", "aten.softmax.int"} <= primitives.WATCHED
    source = Path(primitives.__file__).read_text(encoding="utf-8")
    assert '"actual_argument_"+key' in source
    assert '"attention_fp32_gemm_control"' in source
    assert "query-sliced" in source
    assert "Same eight TEST" in source
    assert "Missing native composite attention operators" in source


def test_qwen_reference_quantizes_effective_operands_before_double_replay():
    source = Path(primitives.__file__).read_text(encoding="utf-8")
    assert "return x.to(torch.bfloat16), *values, captures" in source
    assert "reference = x.double() @ weight.double().T + bias.double()" in source
    assert "for partitions in (2, 4, 8, 16):" in source
    assert "cuda_kernel_names" in source


def test_cpu_import_does_not_require_torch_or_initialize_cuda():
    tree = ast.parse(Path(primitives.__file__).read_text(encoding="utf-8"))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(isinstance(n, ast.Import) and any(x.name == "torch" for x in n.names) for n in imports)


def test_gpu_cooperates_with_existing_measurement_queue_before_output_creation():
    source = Path(primitives.__file__).read_text(encoding="utf-8")
    assert '.temporal_telemetry_a100.lock' in source
    assert 'fcntl.LOCK_EX | fcntl.LOCK_NB' in source
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    outer = next(n for n in main.body if isinstance(n, ast.With))
    assert isinstance(outer.body[0], ast.Expr)
    assert outer.body[0].value.func.id == "run_checked"


def test_diagnostic_non_a100_ignores_other_cards_but_rejects_target_foreign(monkeypatch):
    import pytest
    selected = "GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selected)
    def response(command, **_):
        if "--query-compute-apps=pid,gpu_uuid,process_name" in command:
            return "999, " + primitives.frozen.A100_UUID + ", foreign_timing\n"
        return selected + ", NVIDIA GeForce RTX 4090, 450\n"
    monkeypatch.setattr(primitives.subprocess, "check_output", response)
    assert primitives.device_audit(selected)["name"] == "NVIDIA GeForce RTX 4090"
    monkeypatch.setattr(primitives.subprocess, "check_output", lambda *a, **k: "999, " + selected + ", foreign\n")
    with pytest.raises(RuntimeError, match="occupied"):
        primitives.device_audit(selected)
    with pytest.raises(ValueError, match="Unapproved"):
        primitives.device_audit("GPU-afc19890-6209-ee4d-622d-e619da5bd5b2")


def test_cross_hardware_comparison_is_not_same_gpu_fidelity_guard():
    source = Path(primitives.__file__).read_text(encoding="utf-8")
    assert 'same_gpu_native_observation_max_abs' in source
    assert 'original_a100_prediction_max_abs' in source
    assert 'mode == "native" and gpu_uuid == frozen.A100_UUID' in source
    assert 'physical_gpu' in source
