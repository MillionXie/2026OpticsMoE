"""CPU contracts for scoped rental work; no weights/data/network required."""
import ast
import os
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_temporal as r
from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_assets as a
from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_suite as s

@pytest.mark.parametrize("key,width",[("deepseek",2560),("kimi",2048)])
def test_original_training_budget(key,width):
    c=r.protocol(key)
    assert (c["train"],c["test"],c["validation"],c["epochs"],c["training_batch"],c["lr"],c["seed"])==(2250,558,0,50,512,.001,42)
    assert c["quality_output_trainable_parameters"]==5*width
    assert c["backbone_trainable_parameters"]==0
    assert c["selection"]=="highest_TEST_SRCC_development_no_VAL"
    assert not any(c[k] for k in ("lora","adapters","quantization","generation"))

def test_fixed_four_frames_and_costs():
    for key in a.MODELS:
        c=r.protocol(key)
        assert c["frame_fractions"]==[.1,.37,.63,.9]
        assert c["crop"]==.65 and c["inference_batch"]==1
        assert c["host_proxy_w"]==338.2
        assert c["resident"]==dict(repeats=3,warmup_s=5,measurement_min_s=30,queue_depth=8,poll_ms=50)

def test_pinned_assets_and_no_k25():
    assert set(a.MODELS)=={"deepseek","kimi"}
    assert all(len(revision)==40 for _,revision in a.MODELS.values())
    assert a.MODELS["deepseek"][0]=="deepseek-ai/deepseek-vl2"

def test_rental_deadline_and_gpu_identity():
    assert r.DEADLINE==s.DEADLINE==1792185600
    assert r.GPU==s.GPU=="GPU-780326a6-79bc-7a45-36f6-683c5b278bee"
    assert r.RUN.name=="lgvq_full_deepseek_kimi_rental_20261017"

def test_no_model_patch_power_change_or_graph():
    text=Path(r.__file__).read_text()
    assert "torch.cuda.CUDAGraph" not in text and "torch.compile(" not in text
    assert "nvidia-smi -pl" not in text
    assert "loss.backward()" in text and "model.train()" not in text

def test_readout_and_native_parity_guard():
    text=Path(r.__file__).read_text()
    assert "language.lm_head(hidden.to(language.lm_head.weight.dtype)).float()" in text
    assert "torch.allclose(full,projection,rtol=.01,atol=.125)" in text
    assert "before!=after" in text

def test_kimi_nested_official_backend_is_explicit_and_audited():
    text=Path(r.__file__).read_text()
    assert 'cfg.vision_config._attn_implementation="flash_attention_2"' in text
    assert 'cfg.text_config._attn_implementation="flash_attention_2"' in text
    assert 'actual_attention=attention' in text

def test_receipt_hashes_final_identity_before_artifacts():
    tree=ast.parse(Path(r.__file__).read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
    persisted=[n.lineno for n in ast.walk(train) if isinstance(n,ast.Call)
               and ast.unparse(n.func)=="write" and any(isinstance(c,ast.Constant)
               and c.value=="identity.json" for c in ast.walk(n))]
    artifact=next(n.lineno for n in train.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=="artifacts" for t in n.targets))
    assert max(persisted)==artifact-1

def test_fresh_process_stages():
    text=Path(s.__file__).read_text()
    assert '("cpu-preflight","train","evaluate","utilization")' in text
    assert "subprocess.Popen" in text and "child.kill()" in text
    assert 'if code:break' in text

def test_resident_guard_does_not_block_in_forward_queue():
    text=Path(r.__file__).read_text()
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="utilization")
    phase=next(n for n in ast.walk(function) if isinstance(n,ast.FunctionDef) and n.name=="phase")
    calls=[ast.unparse(n.func) for n in ast.walk(phase) if isinstance(n,ast.Call)]
    assert "guard.check" not in calls and "guard.fast_check" in calls

def test_dynamic_width_head_only_when_torch_available():
    if os.name=="nt":pytest.skip("Native tensor contract is checked on the Linux rental runtime")
    torch=pytest.importorskip("torch")
    for width in (2048,2560):
        initial=torch.randn(5,width)
        head=r.five_rows(initial)
        assert head.weight.numel()==5*width
        head(torch.randn(1,width)).sum().backward()
        assert head.weight.grad is not None and not initial.requires_grad
