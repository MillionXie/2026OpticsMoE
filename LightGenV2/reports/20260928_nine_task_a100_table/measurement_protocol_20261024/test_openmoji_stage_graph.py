"""No-Torch contracts for unchanged dispatch, six passes and CPU-only admission."""
from collections import Counter
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import openmoji_e45_graph as graph


class GoodDispatch:
    def forward(self,input_fields):
        return super().forward(full_aperture_language_amplitude(input_fields))


class BadDispatch:
    def forward(self,input_fields):
        return super().forward(input_fields)


def test_exact_original_subclass_dispatch_is_required():
    graph.full_aperture_dispatch_contract(GoodDispatch.forward)
    with pytest.raises(ValueError):graph.full_aperture_dispatch_contract(BadDispatch.forward)


def test_counts_include_true_bridge_and_each_modality_three_passes():
    expected=graph.validate_stage_counts(Counter({
        'frontend.plain_token_embedding':1,'frontend.image_normalization':1,
        'frontend.patch_and_position':1,'frontend.original_sinusoidal_position':1,
        'bridge.original_prompt_condition_to_vision':1,'head.position_linear_language_summary':1,
        'head.original_shared_grid_readout':1,'language.full_aperture_amplitude_layout':1,
        **{m+'.'+n:v for m in ('language','vision') for n,v in {
            'router_simulation':1,'router_energy_top2':1,'expert_global_simulation':2,
            'original_scalar_ccd_normalization':2,'expert_original_ccd_pool':1,'global_original_ccd_pool':1,
            'parallel_residual1':1,'parallel_residual2':1}.items()}}))
    assert sum(expected[m+'.router_simulation']+expected[m+'.expert_global_simulation'] for m in ('language','vision'))==6
    wrong=Counter(expected);wrong['bridge.original_prompt_condition_to_vision']=0
    with pytest.raises(ValueError):graph.validate_stage_counts(wrong)


def test_no_zero_super_AST_or_global_parent_class_mutation():
    text=(HERE/'openmoji_e45_graph.py').read_text(encoding='utf8')
    assert "target=SimpleNamespace(forward=MethodType(OpticalDetectorTopKRouter.forward,router))" in text
    assert "meter.patch_statements(target,'forward'" in text
    assert 'OpticalDetectorTopKRouter.forward=' not in text
    assert "full_aperture_language_amplitude(input_fields)" in text
    assert "hasattr(core,'output_adapter')" in text


def test_true_online_input_and_reuse_not_cached_producer_retest():
    text=(HERE/'openmoji_e45_graph.py').read_text(encoding='utf8')
    assert 'load_prompt_cache' not in text and 'nn.Identity(' not in text
    assert "tokenizer.encode(record['instruction'],add_special_tokens=False)" in text
    assert "cached_live_frontend_contract_reused_not_retested=True" in text
    assert 'torch.equal(expected[k],actual[k])' in text
    assert "for index in (200,999):" in text
    assert "source_files()!=frozen" in text


def test_cpu_controller_never_creates_cuda_or_mps_and_keeps_shared_lock():
    text=(HERE/'run_openmoji_stage_contract.py').read_text(encoding='utf8')
    assert 'import torch' not in text and 'engine.daemon(' not in text
    assert "CUDA_VISIBLE_DEVICES=''" in text
    assert 'fcntl.LOCK_EX|fcntl.LOCK_NB' in text
    assert 'Foreign GPU appeared during CPU stage contract' in text


def test_shape_tracing_reads_metadata_only():
    class FakeTensor:shape=(1,192,14,14);dtype='float32'
    assert graph.tensor_shapes({'spatial':FakeTensor()})=={'spatial':dict(shape=[1,192,14,14],dtype='float32')}
