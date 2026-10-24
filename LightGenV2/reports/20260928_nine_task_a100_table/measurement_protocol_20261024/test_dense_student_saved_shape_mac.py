import io
from pathlib import Path
import pickle
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_student_saved_shape_mac import DenseMetadataUnpickler,inert_posix_path,nn_terms


def test_metadata_paths_are_inert_and_TorchVersion_is_text_only():
    reader=DenseMetadataUnpickler(io.BytesIO())
    assert reader.find_class('pathlib','PosixPath')('/','data','x')==inert_posix_path('/','data','x')
    assert reader.find_class('torch.torch_version','TorchVersion')('2.6.0')=='2.6.0'
    assert 'torch' not in sys.modules
    with pytest.raises(ValueError):reader.find_class('os','system')


def test_original_primitives_stay_primitives():
    v=dict(core={},head={},epoch=6)
    assert DenseMetadataUnpickler(io.BytesIO(pickle.dumps(v,protocol=2))).load()==v


def matrices(task):
    core={'hybrid.input_adapter.weight':[192,1024],'hybrid.output_adapter.weight':[1024,192],
        'hybrid.optical_branch.core.input_adapter.weight':[224,192],
        'hybrid.optical_branch.core.output_adapter.weight':[192,224],'hybrid.optical_branch.expert_output_adapter.weight':[192,224]}
    for i in (0,1):
        prefix=f'hybrid.blocks.{i}.'
        core.update({prefix+'token_depthwise.weight':[192,1,3,3],prefix+'token_pointwise.weight':[192,192],
            prefix+'mlp.0.weight':[384,192],prefix+'mlp.3.weight':[192,384]})
        if task=='SALICON':core[prefix+'mlp.1.0.conv.weight']=[384,1,3,3]
    current=160 if task=='LSP' else 128;head={'token_projection.weight':[current,192]}
    for i in (0,1):head.update({f'body.{i}.block.0.weight':[current,1,3,3],f'body.{i}.block.1.weight':[current,current,1,1]})
    for i,after in enumerate((128,96) if task=='LSP' else (96,64,32,16)):
        head.update({f'upsample.{i}.block.0.weight':[current,1,3,3],f'upsample.{i}.block.1.weight':[after,current,1,1]});current=after
    head.update({'refine.block.0.weight':[current,1,3,3],'refine.block.1.weight':[current,current,1,1],
        ('predictor' if task=='LSP' else 'classifier')+'.weight':[14 if task=='LSP' else 1,current,1,1]})
    return {group:{k:dict(__tensor_descriptor__=True,shape=v) for k,v in values.items()} for group,values in [('core',core),('head',head)]}


def test_one_actual_merger_and_progressive_pose_not_teacher_deconv():
    v=nn_terms('LSP',matrices('LSP'))
    assert v['serial_NN_terms']['native_one_main_merger49']==49*(4096**2+4096*2048)
    assert v['serial_NN_terms']['head.up28.pointwise']==28**2*160*128
    assert v['serial_NN_terms']['head.output']==56**2*96*14
    assert v['actual_NN_MAC']==v['charged_serial_NN_MAC']+v['parallel_residual_NN_MAC']


def test_saliency_expanded_spatial_FFN_is_real_residual_cost():
    v=nn_terms('SALICON',matrices('SALICON'))
    assert v['parallel_NN_terms']['hybrid.blocks.0.expanded_CFFN_DW3']==196*384*9
    assert v['serial_NN_terms']['head.output']==224**2*16
    assert v['other_executed_digital_dense_MAC']['GPU_router_dense_mask_contraction']==4*478**2


def test_no_missing_or_extra_NN_matrix():
    maps=matrices('LSP');maps['core']['invented.weight']=dict(__tensor_descriptor__=True,shape=[192,192])
    with pytest.raises(ValueError):nn_terms('LSP',maps)
