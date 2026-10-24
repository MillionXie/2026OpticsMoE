import io
from pathlib import Path
import pickle
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_student_saved_shape_mac import DenseMetadataUnpickler,inert_posix_path


def test_metadata_paths_are_inert_and_TorchVersion_is_text_only():
    reader=DenseMetadataUnpickler(io.BytesIO())
    assert reader.find_class('pathlib','PosixPath')('/','data','x')==inert_posix_path('/','data','x')
    assert reader.find_class('torch.torch_version','TorchVersion')('2.6.0')=='2.6.0'
    assert 'torch' not in sys.modules
    with pytest.raises(ValueError):reader.find_class('os','system')


def test_original_primitives_stay_primitives():
    v=dict(core={},head={},epoch=6)
    assert DenseMetadataUnpickler(io.BytesIO(pickle.dumps(v,protocol=2))).load()==v
