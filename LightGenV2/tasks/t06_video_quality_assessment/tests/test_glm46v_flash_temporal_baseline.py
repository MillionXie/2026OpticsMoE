import ast
from pathlib import Path

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import glm46v_flash_temporal_baseline as run
from LightGenV2.tasks.t06_video_quality_assessment.prepare_vlm_assets import validate_profile


def source(name):
    tree = ast.parse(Path(run.__file__).read_text())
    return ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def test_original_protocol_except_native_hidden_width():
    c = run.config()
    assert c['frames'] == 4 and c['epochs'] == 50 and c['inference_batches'] == [1]
    assert c['thinking'] is False and c['hidden_width'] * 5 == 20480
    assert c['train_samples'] == 2250 and c['test_samples'] == 558
    assert c['backbone_trainable_parameters'] == 0
    assert c['selection'] == 'highest_TEST_SRCC_development_no_VAL'


def test_pinned_public_assets():
    c = run.config()
    validate_profile(c)
    assert c['model_revision'] == '411bb4d77144a3f03accbf4b780f5acb8b7cde4e'
    assert len(c['official_assets']) == 13
    assert sum(v['size'] for k, v in c['official_assets'].items() if k.endswith('.safetensors')) == 20585644048


def test_source_binding_and_finite_suite():
    assert all(len(v) == 64 for v in run.sources().values())
    s = source('suite')
    assert '10800' in s and '7200' in s and 'no automatic retry' in s
    assert 'a100_batch04' not in s and 'a100_batch08' not in s


def test_native_backbone_no_thinking_generation_or_patch():
    s = source('native_model')
    assert 'Glm4vForConditionalGeneration.from_pretrained' in s
    assert 'enable_thinking=False' in s and 'requires_grad_(False)' in s
    assert "'5.0.0'" in s and 'bfloat16' in s
    f = source('features')
    assert 'model.model(**inputs' in f and 'last_hidden_state' in f
    assert '.generate(' not in f


def test_four_frames_geometry_metadata_and_no_resampling():
    s = source('prepare_batch')
    assert 'do_sample_frames=False' in s and "'video_metadata'" in s
    assert '448 * 448' in s and 'len(d[0]) != 4' in s
    assert 'core.FRAME_FRACTIONS[4]' in s


def test_original_single_loss_optimizer_no_scan():
    s = source('train_rows')
    assert 'range(1, 51)' in s and 'cross_entropy' in s
    assert 'lr=0.001, weight_decay=0.0' in s and 'T_max=50' in s
    assert 'FiveNativeTokenRows(initial)' in s
    assert '20480' in s and 'strict_cpu_metrics' in s


def test_accuracy_only_and_fresh_full_test():
    s = source('run')
    assert 'for i, row in enumerate(test)' in s
    assert 'formal_latency_energy_measured=False' in s and 'test_videos=558' in s
    assert 'runtime_sha256' in s and 'training_receipt.json' in s
    assert 'native_forward_last_logits_vs_feature_tap_max_abs' in s


def test_admission_not_small_model_threshold():
    assert '32000' in source('guard')


def test_refuse_old_outputs_before_loading(tmp_path):
    with pytest.raises(FileExistsError):
        run.shared.new_output(tmp_path)
