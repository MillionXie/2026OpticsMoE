"""Bounded, read-only PyTorch ZIP metadata without Torch or tensor storage IO.

The first local ZIP entry of an ordinary torch.save archive is data.pkl. Read
only that prefix, so an archive nested inside an old compressed lab package
does not need its multi-GiB tensor payload decompressed/searched to EOF.
Only primitive containers, storage descriptors and inert tensor descriptors
are allowed. No arbitrary pickle globals or model classes are executed.
This is a metadata identity aid, NOT a tensor/content/numerical equivalence.
"""
from __future__ import annotations
from collections import OrderedDict
import io
import pickle
import pickletools
import struct


def first_pickle(stream,limit=4*1024*1024):
    if limit<128:raise ValueError('Metadata byte limit too small')
    header=stream.read(30)
    if len(header)!=30:raise ValueError('Missing first local ZIP header')
    fields=struct.unpack('<IHHHHHIIIHH',header)
    signature,_,flags,method,_,_,_,size,_,name_size,extra_size=fields
    if signature!=0x04034b50 or method!=0 or flags&1:
        raise ValueError('Only unencrypted ZIP_STORED first entry supported')
    name=stream.read(name_size)
    extra=stream.read(extra_size)
    if len(name)!=name_size or len(extra)!=extra_size or not name.endswith(b'/data.pkl'):
        raise ValueError('First local entry is not the original data.pkl')
    data=bytearray()
    if size not in (0,0xffffffff) and not flags&8:
        if size>limit:raise ValueError('Metadata exceeds read bound')
        data.extend(stream.read(size))
        if len(data)!=size:raise ValueError('Truncated metadata')
    else:
        while len(data)<limit:
            block=stream.read(min(65536,limit-len(data)))
            if not block:break
            data.extend(block)
            try:
                last=next((position for opcode,_,position in pickletools.genops(data) if opcode.name=='STOP'),None)
            except ValueError:continue
            if last is not None:return bytes(data[:last+1])
        raise ValueError('No complete bounded pickle prefix')
    try:
        stop=next((position for opcode,_,position in pickletools.genops(data) if opcode.name=='STOP'),None)
    except ValueError as error:raise ValueError('Invalid metadata pickle') from error
    if stop is None:raise ValueError('No complete bounded pickle prefix')
    return bytes(data[:stop+1])


def tensor_descriptor(storage,offset,size,stride,*unused):
    if not isinstance(storage,dict) or storage.get('__storage_descriptor__') is not True:
        raise ValueError('Invalid storage reference')
    return dict(__tensor_descriptor__=True,dtype=storage['dtype'],storage_key=storage['key'],
        storage_numel=storage['numel'],storage_location=storage['location'],
        storage_offset=int(offset),shape=list(size),stride=list(stride))


def parameter_descriptor(data,*unused):
    if not isinstance(data,dict) or not data.get('__tensor_descriptor__'):
        raise ValueError('Invalid parameter descriptor')
    return dict(data,parameter=True)


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='collections' and name=='OrderedDict':return OrderedDict
        if module=='torch' and name in ('HalfStorage','FloatStorage','DoubleStorage','BFloat16Storage',
            'LongStorage','IntStorage','ShortStorage','CharStorage','ByteStorage','BoolStorage',
            'ComplexFloatStorage','ComplexDoubleStorage'):
            return 'torch.'+name
        if module=='torch._utils' and name in ('_rebuild_tensor','_rebuild_tensor_v2'):
            return tensor_descriptor
        if module=='torch._utils' and name=='_rebuild_parameter':return parameter_descriptor
        raise ValueError('Unsupported pickle global, not executed: '+module+'.'+name)

    def persistent_load(self,identity):
        if (not isinstance(identity,tuple) or len(identity)!=5 or identity[0]!='storage'
                or not isinstance(identity[1],str) or not identity[1].startswith('torch.')
                or not isinstance(identity[4],int) or identity[4]<0):
            raise ValueError('Unsupported storage descriptor')
        return dict(__storage_descriptor__=True,dtype=identity[1],key=str(identity[2]),
            location=str(identity[3]),numel=identity[4])


def read_metadata(stream,limit=4*1024*1024):
    raw=first_pickle(stream,limit)
    value=MetadataUnpickler(io.BytesIO(raw)).load()
    if not isinstance(value,dict):raise ValueError('Expected original cache metadata mapping')
    return value,raw
