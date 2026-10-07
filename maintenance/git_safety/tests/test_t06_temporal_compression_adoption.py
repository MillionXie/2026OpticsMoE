"""Synthetic CPU checks only; never uses recorded video/cache/PT or hardware."""
import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from LightGenV2.tasks.t06_video_quality_assessment import compress_temporal_readout as module
from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import load_settings

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'LightGenV2/tasks/t06_video_quality_assessment'


def test_original_scientific_functions_are_unchanged_except_output_guard():
    evidence = json.loads((BASE / 'temporal_compression_import_20261007.json').read_text(encoding='utf8'))
    tree = ast.parse((BASE / 'compress_temporal_readout.py').read_text(encoding='utf8'))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, digest in evidence['original_function_AST_SHA256'].items():
        node = functions[name]
        if name in ('extract', 'train'):
            first = node.body.pop(0)
            assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            assert isinstance(first.value.func, ast.Name) and first.value.func.id == '_reject_existing_outputs'
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == digest


@pytest.mark.parametrize('existing', ['cache', 'report'])
def test_extract_refuses_existing_outputs_before_reading_inputs(tmp_path, existing):
    output = tmp_path / 'cached.pt'
    target = output if existing == 'cache' else output.with_suffix('.json')
    target.write_bytes(b'preserved')
    with patch.object(module, 'load_settings', side_effect=AssertionError('must not load')):
        with pytest.raises(FileExistsError):
            module.extract(config=tmp_path / 'missing.yaml', checkpoint=tmp_path / 'missing.pt',
                           output=output, device='cpu', physical_batch_size=1, train_grouping_views=1)
    assert target.read_bytes() == b'preserved'


def test_train_refuses_existing_run_before_rng_cuda_or_loading(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    with patch.object(module.random, 'seed', side_effect=AssertionError('must not change RNG')):
        with pytest.raises(FileExistsError):
            module.train(config=tmp_path / 'missing.yaml', cache_path=tmp_path / 'missing_cache.pt',
                         source_checkpoint=tmp_path / 'missing_source.pt', output_dir=run,
                         device='cpu', epochs=1, batch_size=1, learning_rate=.001,
                         weight_decay=0., regression_weight=1., ranking_weight=0.,
                         correlation_weight=0., teacher_weight=0., ema_decay=0., mos_strata=0, seed=1)
    assert list(run.iterdir()) == []


@pytest.mark.parametrize('width,seed', [(256, 171), (512, 170), (640, 172)])
def test_optional_profiles_keep_16x4_and_only_narrow_final_head(width, seed):
    settings = load_settings(BASE / f'configs/lightgen/temporal_multivideo16x4_readout_h{width}_s{seed}.yaml')
    assert (settings.videos_per_field, settings.frame_count) == (16, 4)
    assert settings.temporal_readout_mode == 'pruned'
    assert settings.temporal_readout_hidden_width == width


def test_batches_only_return_explicit_fit_indices():
    cache = {key: torch.arange(4).float().reshape(4, 1) for key in
             ('vision', 'language', 'mask', 'target_normalized', 'normalized_prediction')}
    batches = list(module._batches(cache, torch.tensor([0, 2]), batch_size=1,
                                  generator=None, mos_strata=0))
    assert torch.cat([row[-1] for row in batches]).tolist() == [0, 2]


@pytest.mark.parametrize('rank,seed', [(512,173),(384,174),(320,175),(256,176),
                                      (192,177),(128,178),(64,179),(48,180)])
def test_optional_low_rank_configs_preserve_original_bytes_and_geometry(rank, seed):
    evidence = json.loads((BASE / 'temporal_compression_import_20261007.json').read_text())
    rows = {r['name']: r for r in evidence['additional_low_rank_configs']['files']}
    name = f'temporal_multivideo16x4_readout_rank{rank}_s{seed}.yaml'
    path = BASE / 'configs/lightgen' / name
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == rows[name]['sha256_lf']
    settings = load_settings(path)
    assert (settings.videos_per_field, settings.frame_count) == (16, 4)
    assert settings.temporal_readout_mode == 'low_rank'
    assert settings.temporal_readout_rank == rank
    suffix = f'multivideo16x4_readout_rank{rank}_s{seed}' if rank != 48 else 'multivideo16x4_readout_rank48_kd_s180'
    assert str(settings.output_dir).replace('\\','/').endswith(suffix)


def test_pruning_selects_original_neurons_not_an_extra_branch():
    readout = nn.Module()
    readout.output = nn.Sequential(nn.Identity(), nn.Linear(3, 2), nn.Identity(),
                                   nn.Identity(), nn.Linear(2, 1))
    source = {'output.1.weight': torch.arange(12).float().reshape(4, 3),
              'output.1.bias': torch.arange(4).float(),
              'output.4.weight': torch.tensor([[1., 2., 3., 4.]]),
              'output.4.bias': torch.tensor([.5])}
    report = module._initialize_pruned_readout(readout, source)
    assert report['retained_hidden_width'] == 2
    assert torch.equal(readout.output[1].weight, source['output.1.weight'][[3, 2]])
    assert torch.equal(readout.output[4].weight, source['output.4.weight'][:, [3, 2]])
