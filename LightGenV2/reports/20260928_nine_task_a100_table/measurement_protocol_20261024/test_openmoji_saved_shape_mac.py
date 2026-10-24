from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from openmoji_saved_shape_mac import module_mac


def row(kind,inputs,outputs,name='model.layer'):
    return dict(module=name,class_name=kind,inputs=[dict(shape=inputs)],outputs=dict(shape=outputs))


def test_actual_linear_conv1d_conv3d_shapes_not_parameter_shortcut():
    assert module_mac(row('Linear',[1,7,2048],[1,7,192]),{'layer.weight':dict(shape=[192,2048])})==7*2048*192
    assert module_mac(row('Conv1d',[1,192,11],[1,192,7]),{'layer.weight':dict(shape=[192,1,5])})==7*192*5
    assert module_mac(row('Conv3d',[1,3,2,224,224],[1,1024,1,14,14]),{'layer.weight':dict(shape=[1024,3,2,16,16])})==196*1024*1536


def test_position_contraction_counts_padded_rows_not_just_valid_length():
    assert module_mac(row('PositionReadout',[],[1,192]),{'layer.weight':dict(shape=[64])})==64*192
    assert module_mac(row('Embedding',[7],[7,2048],name='frontend_embedding'),{}) is None


def test_unknown_is_not_zero():
    with pytest.raises(ValueError):module_mac(row('Unknown',[1],[1]),{'layer.weight':dict(shape=[1])})
