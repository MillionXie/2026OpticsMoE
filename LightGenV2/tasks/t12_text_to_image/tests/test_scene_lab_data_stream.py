import io,zlib,hashlib
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_data_stream import packet,read_packet,write_data


def test_packet_exact_lossless_bytes_and_checksum():
    value=b'fixture-not-CCD'*1000;encoded=zlib.compress(value,1);stream=io.BytesIO();packet(stream,dict(files=[dict(name='raw.npy',bytes=len(value),encoded_bytes=len(encoded),SHA256=hashlib.sha256(value).hexdigest(),codec='zlib')]),[encoded]);stream.seek(0)
    header,files=read_packet(stream);assert files['raw.npy']==value
    with pytest.raises(EOFError):read_packet(io.BytesIO(stream.getvalue()[:-1]))


def test_interrupted_prefix_preserved_and_completed_data_never_changed(tmp_path):
    p=tmp_path/'data.npy';ready=tmp_path/'capture.ready';p.write_bytes(b'prefix')
    write_data(p,b'prefix-rest',ready);assert p.read_bytes()==b'prefix-rest';assert len(list(tmp_path.glob('*.partial_transport_*')))==1
    ready.touch()
    with pytest.raises(ValueError):write_data(p,b'different',ready)
    write_data(p,b'prefix-rest',ready)
