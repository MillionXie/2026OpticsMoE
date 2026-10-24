"""Inert metadata parsing; no Torch import, source model or tensor payload."""
import io
import pickle
from pathlib import Path
import sys
import zipfile
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from torch_metadata import first_pickle,read_metadata,MetadataUnpickler,tensor_descriptor


def archive(payload):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('original/data.pkl',payload)
        z.writestr('original/data/0',b'X'*1000000)
    output.seek(0)
    return output


def test_only_metadata_no_torch_storage_read():
    payload=pickle.dumps(dict(contract='test',source_checkpoint='/original/model.pt'),protocol=2)
    stream=archive(payload);value,actual=read_metadata(stream)
    assert value==dict(contract='test',source_checkpoint='/original/model.pt')
    assert actual==payload and stream.tell()<1024


def test_reject_arbitrary_global_not_execute_it():
    with pytest.raises(ValueError,match='not executed'):
        read_metadata(archive(b'cbuiltins\neval\n(S"1+2"\ntR.'))


def test_explicit_metadata_bound_and_first_member():
    with pytest.raises(ValueError):first_pickle(io.BytesIO(b'not a torch archive'))
    with pytest.raises(ValueError):read_metadata(archive(pickle.dumps(dict(big='a'*10000))),limit=1024)


def test_inert_tensor_shape_no_materialized_tensor():
    loader=MetadataUnpickler(io.BytesIO())
    storage=loader.persistent_load(('storage','torch.HalfStorage','0','cpu',2808*4*196*192))
    value=tensor_descriptor(storage,0,(2808,4,196,192),(4*196*192,196*192,192,1))
    assert value['shape']==[2808,4,196,192] and value['dtype']=='torch.HalfStorage'
    assert value['__tensor_descriptor__'] and 'tensor' not in value
    with pytest.raises(ValueError):loader.find_class('os','system')
    with pytest.raises(ValueError):loader.persistent_load(('not-storage',))
