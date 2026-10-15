import ast
from pathlib import Path

from LightGenV2.tasks.t06_video_quality_assessment import ovis25_temporal_baseline as run


def function_source(name):
    tree=ast.parse(Path(run.__file__).read_text())
    return ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_same_original_training_budget():
    c=run.config()
    assert c["frames"]==4 and c["epochs"]==50 and c["inference_batches"]==[1]
    assert c["thinking"] is False and c["hidden_width"]*5==10240
    assert c["backbone_trainable_parameters"]==0


def test_sources_bind_official_new_adapter_and_unchanged_dependencies():
    assert len(run.sources())==11
    assert all(len(v)==64 for v in run.sources().values())


def test_native_video_does_not_expand_frame_resolution_or_enable_thinking():
    s=function_source("prepare_batch")
    assert "min_pixels=448 * 448, max_pixels=448 * 448" in s
    assert "enable_thinking=False" in s and "[[4, 28, 28]]" in s
    assert "784" in s and "3136, 768" in s


def test_native_feature_tap_uses_original_visual_merge_and_language_backbone():
    s=function_source("features")
    assert "model.merge_multimodal" in s and "model.llm.model" in s
    assert "generate" not in s and "last_hidden_state" in s


def test_five_rows_plain_ce_original_optimizer_scheduler_no_search():
    s=function_source("train_rows")
    assert "range(1, 51)" in s and "cross_entropy" in s
    assert "lr=0.001, weight_decay=0.0" in s and "T_max=50" in s
    assert "FiveNativeTokenRows(initial)" in s


def test_suite_single_B1_and_finite_time_no_retry():
    s=function_source("suite")
    assert "inference_batches=[1]" in s and "10800" in s
    assert "7200" in s and "no automatic retry" in s
    assert "'a100_batch04'" not in s and "'a100_batch08'" not in s
