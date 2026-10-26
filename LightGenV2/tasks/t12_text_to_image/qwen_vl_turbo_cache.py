"""Bounded cache storage helpers; no change to model inputs or output graph."""
import hashlib
from pathlib import Path

import torch


class HashingWriter:
    """torch.save supports a sequential write/flush stream."""
    def __init__(self, handle):
        self.handle=handle
        self.digest=hashlib.sha256()

    def write(self, data):
        count=self.handle.write(data)
        self.digest.update(memoryview(data)[:count])
        return count

    def flush(self):
        self.handle.flush()


def save_with_digest(payload,path: Path):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        writer=HashingWriter(handle)
        torch.save(payload,writer)
        digest=writer.digest.hexdigest()
    temporary.replace(path)
    return digest


def image_float(value,device=None,dtype=torch.float32):
    value=value.to(device=device)
    if value.dtype==torch.uint8:
        value=value.float().div(127.5).sub(1)
    return value.to(dtype=dtype)
