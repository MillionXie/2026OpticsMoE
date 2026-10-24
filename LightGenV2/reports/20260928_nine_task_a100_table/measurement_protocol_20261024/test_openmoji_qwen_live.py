"""Native contract metadata checks without importing Torch or starting CUDA."""
import copy
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import openmoji_qwen_live as live


def values():
    return (dict(qwen_target=.812,qwen_epoch=30,layout_version='layered_anchor6_svg_v3'),
        dict(lightgen_model_variant='qwen_frozen_shared_readout',embedding_only=False,qwen_shared_baseline=True,
             image_size=224,grid_size=6,amp_enabled=True),
        dict(selected_epoch=30,test_samples=1000,metrics=dict(overall=dict(changed_cell_accuracy=.812))))


def test_adopted_same_layout_native_not_cache_placeholder():live.validate_identity(*values())


@pytest.mark.parametrize('section,key,value',[(0,'qwen_epoch',45),(0,'layout_version','grid_v2'),
    (1,'embedding_only',True),(1,'image_size',448),(1,'amp_enabled',False),(2,'test_samples',200)])
def test_reject_wrong_identity_or_workload(section,key,value):
    args=list(copy.deepcopy(values()));args[section][key]=value
    with pytest.raises(ValueError):live.validate_identity(*args)


def test_wrapper_does_not_replace_original_native_calls():
    source=(HERE/'openmoji_qwen_live.py').read_text()
    assert 'extract_native_one(native,observed,row,settings,device)' in source
    assert "head(visual.unsqueeze(0),[language])" in source
    assert 'for index in (200,999)' in source and 'torch.cuda.is_initialized()' in source
    assert "min_pixels=224**2,max_pixels=224**2" in source
    assert "Identity(" not in source


def test_controller_has_no_torch_import_and_stops_only_own_child():
    source=(HERE/'run_openmoji_qwen_contract.py').read_text()
    assert 'import torch' not in source and "CUDA_VISIBLE_DEVICES=''" in source
    assert "fcntl.LOCK_EX|fcntl.LOCK_NB" in source and 'while idle<3' in source
    assert 'child.terminate()' in source and 'os.kill(' not in source
