import ast
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import diagnose_batch_layers as trace


def test_bounded_samples_and_cpu_reference_memory():
    assert trace.BATCHES==(1,2,4)
    assert trace.REFERENCE_BYTE_LIMIT==8*1024**3


def test_no_training_or_model_forward_patch():
    tree=ast.parse(Path(trace.__file__).read_text(encoding="utf-8"))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {"backward","step","train"}.intersection(calls)
    assert not any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and n.attr=="forward" for n in ast.walk(tree))


def test_reference_stage_report_handles_discrete_routes():
    t=trace.Trace(); t.events=["v","gate"]
    t.statistics={"v":dict(kind="vision",max_abs=2.,square_sum=4.,reference_square_sum=16.,entries=4,changed_entries=1),
                  "gate":dict(kind="route",tokens=20,changed_token_routes=2)}
    assert t.report()["v"]["relative_l2"]==.5
    assert t.report()["gate"]["changed_token_routes"]==2


def test_suite_is_bounded_no_occupancy_queue_and_preserves_original_data():
    from LightGenV2.tasks.t06_video_quality_assessment import run_batch_layer_diagnostics as suite
    assert suite.CHILD_TIMEOUT_SECONDS==600
    tree=ast.parse(Path(suite.__file__).read_text(encoding="utf-8"))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {"sleep","backward","train","cuda"}.intersection(calls)
    assert "do not retry blindly" in Path(suite.__file__).read_text(encoding="utf-8")


def test_direct_diagnosis_detaches_uninitialized_formal_timer_hooks():
    class Timer:
        attached = True
        def close(self):
            self.attached = False
        def forward(self):
            if self.attached:
                raise KeyError("vision_calls")
            return "native_forward"
    class Handle:
        removed = False
        def remove(self):
            self.removed = True
    timer, handle = Timer(), Handle()
    handles = [handle]
    with pytest.raises(KeyError, match="vision_calls"):
        timer.forward()
    trace.detach_cost_instrumentation(timer, handles)
    assert timer.forward() == "native_forward"
    assert handle.removed and handles == []


def test_timer_detached_before_trace_hooks_and_native_forward():
    source = Path(trace.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    gpu = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gpu_checks")
    calls = [n for n in ast.walk(gpu) if isinstance(n, ast.Call)]
    detached = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "detach_cost_instrumentation")
    installed = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "install_hooks")
    forwarded = next(n.lineno for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "diagnostic_forward")
    assert detached < installed < forwarded


def test_native_keyword_hidden_state_hook_contract(monkeypatch):
    marker = object()
    monkeypatch.setattr(trace,"first_tensor",lambda value: marker if value is marker else None)
    class Module:
        def register_forward_pre_hook(self, hook, **options):
            assert options == {"with_kwargs": True}
            self.before = hook
            return object()
        def register_forward_hook(self, hook):
            self.after = hook
            return object()
    decoder = type("Qwen3VLTextDecoderLayer",(Module,),{})()
    class Model:
        def named_modules(self):
            return [("model.language_model.layers.0",decoder)]
    class Recorder:
        def __init__(self): self.events=[]
        def capture(self,*args): self.events.append(args)
    recorder = Recorder()
    trace.install_hooks(Model(),recorder)
    decoder.before(decoder,(),{"position_embeddings":object(),"hidden_states":marker})
    assert recorder.events == [("model.language_model.layers.0:input",marker,"language")]


def test_input_fingerprint_precedes_inplace_transfer(monkeypatch):
    cpu={"pixels":"fp32"}
    monkeypatch.setattr(trace.previous,"input_records",lambda value,count:[value["pixels"]])
    def transfer(value):
        value["pixels"]="bf16"
        return value
    assert trace.verified_transfer(cpu,1,["fp32"],transfer) is cpu
    assert cpu["pixels"] == "bf16"
    original={"pixels":"changed"}
    with pytest.raises(ValueError,match="Traced inputs"):
        trace.verified_transfer(original,1,["fp32"],transfer)
    assert original["pixels"] == "changed"


def test_focus_capture_is_persisted_per_batch_and_model_order_is_bound():
    source=Path(trace.__file__).read_text(encoding="utf-8")
    assert 'trace.snapshots = {}' in source
    assert 'focused_boundaries.pt' in source
    assert 'merger_component' in source and '".linear_fc1"' in source
    from LightGenV2.tasks.t06_video_quality_assessment import run_batch_layer_diagnostics as suite
    runner=Path(suite.__file__).read_text(encoding="utf-8")
    assert 'Baseline selection/order differs from CPU receipt' in runner
    assert 'for name in baselines:' in runner
