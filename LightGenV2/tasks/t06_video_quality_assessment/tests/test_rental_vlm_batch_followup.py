import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_batch_followup as f


def test_original_source_and_protocol_untouched():
    assert f.old.protocol("deepseek")["inference_batch"] == 1
    assert f.old.WIDTHS == {"deepseek":2560,"kimi":2048}
    assert len(f.sources()) == len(f.old.sources()) + 1


def test_real_batch_native_processing_and_no_training():
    text=Path(f.__file__).read_text()
    assert 'processor.batchify(prepared)' in text
    assert 'text=[prompt]*len(selected)' in text
    assert 'test[:2]' in text and 'range(0,558,2)' in text
    assert '.backward(' not in text and 'torch.compile(' not in text
    assert 'range(1,4)' in text and 'phase("warmup",5)' in text
    assert 'phase("measured",30)' in text and 'range(8)' in text


def test_independent_resident_and_dataset_cost():
    text=Path(f.__file__).read_text()
    assert 'costs_not_derived_from_resident_power=True' in text
    assert 'telemetry.workload_cost(cost,power["active_mean_w"])' in text
    assert 'receipt["identity"]["source_sha256"] != old.sources()' in text
    assert 'time.time() >= self.args.deadline' in text
    ast.parse(text)
