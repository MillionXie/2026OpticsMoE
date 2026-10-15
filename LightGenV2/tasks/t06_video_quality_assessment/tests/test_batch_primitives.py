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
