import pytest
from LightGenV2.tasks.t06_video_quality_assessment import batch_telemetry as audit


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8, 9, 10])
def test_plan_preserves_all_ids_and_exact_cost(batch):
    plan = audit.inference_plan(558, batch)
    assert [i for start, end in plan for i in range(start, end)] == list(range(558))
    cost = [(start, end) for start, end in plan if end <= 200]
    assert sum(end-start for start, end in cost) == 200
    assert all(end-start <= batch for start, end in plan)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_normalization_matches_old_full_batch_formula(batch):
    rows = [[0, batch, True, 10*batch] for _ in range(200//batch)]
    cost = audit.workload_cost(rows, 50)
    assert cost["latency_16_videos_ms"] == pytest.approx(160)
    assert cost["energy_16_videos_j"] == pytest.approx(388.2*.160)


def test_utilization_not_power_or_memory():
    samples = [audit.Sample(0, 0, "active:0", 42, 17, 2000),
               audit.Sample(1, 1, "between_calls", 99, 100, 9000)]
    result = audit.telemetry_report(samples)
    assert result["mean_gpu_utilization_percent"] == 17
    assert result["mean_board_power_w"] == 42
    assert result["peak_memory_used_mib"] == 2000


def test_missing_utilization_is_not_zero():
    with pytest.raises(RuntimeError):
        audit.telemetry_report([])


def test_suite_reuses_pt_and_bounds_extra_batches(tmp_path):
    from LightGenV2.tasks.t06_video_quality_assessment import run_temporal_telemetry_suite as suite
    command = suite.child_command("yolo", 3, tmp_path / "manifest", tmp_path / "run")
    assert "--telemetry" in command and "--reproduction-receipt" in command
    assert "--suite" not in command
    assert "evaluate_temporal_baseline_batches" in " ".join(command)
    assert "best_checkpoint.pt" in " ".join(command)


def test_diagnostic_has_no_weight_training():
    import ast
    from pathlib import Path
    from LightGenV2.tasks.t06_video_quality_assessment import diagnose_batch_numerics as diagnosis
    tree = ast.parse(Path(diagnosis.__file__).read_text(encoding="utf-8"))
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"backward", "step", "train"}.intersection(calls)


def test_diagnostic_deepseek_dict_output_uses_explicit_keys(monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    from LightGenV2.tasks.t06_video_quality_assessment import diagnose_batch_numerics as diagnosis

    class Tensor(np.ndarray):
        def bool(self):
            return self.astype(bool)

    def tensor(values):
        return np.asarray(values).view(Tensor)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor))

    class DictOutput:
        def __init__(self, values):
            self.values = values
        def keys(self):
            return self.values.keys()
        def __getitem__(self, key):
            return self.values[key]

    values = dict(input_ids=tensor([[1, 2, 0], [1, 2, 0]]),
                  attention_mask=tensor([[1, 1, 0], [1, 1, 0]]),
                  images=tensor(np.zeros((2, 4, 3, 2, 2))),
                  images_spatial_crop=tensor(np.ones((2, 4, 2))),
                  images_seq_mask=tensor([[0, 1, 0], [0, 1, 0]]))
    monkeypatch.setattr(diagnosis, "tensor_hash", lambda tensor: str(tensor.tolist()))
    records = diagnosis.input_records(DictOutput(values), 2)
    assert len(records) == 2 and records[0] == records[1]
    assert records[0]["input_ids"] == "[1, 2]"
    assert records[0]["images_seq_mask"] == "[0, 1]"


def test_reservation_ignores_other_cards_but_not_foreign_a100():
    from LightGenV2.tasks.t06_video_quality_assessment import reserve_temporal_a100 as reservation
    records = [{"pid": "1", "gpu_uuid": reservation.frozen.A100_UUID},
               {"pid": "2", "gpu_uuid": reservation.frozen.A100_UUID},
               {"pid": "3", "gpu_uuid": "other_gpu"}]
    assert reservation.foreign_processes(records, 1) == [records[1]]
    assert reservation.RESERVATION_BYTES == 64*1024*1024
    assert reservation.MIN_FREE_BYTES >= 2*1024*1024*1024
