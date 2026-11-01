"""Explicit per-process GPU selection; legacy runs retain their original default."""
import os,re

LEGACY_GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def configured_gpu(environ=None):
    value=(os.environ if environ is None else environ).get('T12_GPU_UUID',LEGACY_GPU)
    if not isinstance(value,str)or not re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',value):
        raise ValueError('T12_GPU_UUID must be an explicit complete GPU UUID')
    return value
