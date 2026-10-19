"""One-frame bounded PNG persistence; no camera/SLM/model calls in the writer.

The caller owns acquisition and receipt order. It must finish the previous item
after capturing the next frame and drain at every role/error/exit boundary.
Arrays are copied before scheduling so an SDK buffer cannot be reused under IO.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image


def write_images(base,raw,crop):
    base=Path(base);raw_path=base.with_name(base.name+'_raw.png');crop_path=base.with_suffix('.png')
    assert not raw_path.exists() and not crop_path.exists()
    Image.fromarray(raw).save(raw_path,compress_level=1)
    Image.fromarray(crop).save(crop_path,compress_level=1)
    digest=hashlib.sha256()
    with raw_path.open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):digest.update(part)
    return dict(raw_SHA256=digest.hexdigest(),raw_file=str(raw_path),crop_file=str(crop_path),persisted_at_ns=time.time_ns())


class FrameWriter:
    """At most one immutable frame snapshot outstanding; explicit receipt drain."""
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='DVP_PNG_only');self.pending=None;self.closed=False

    def submit(self,base,raw,crop):
        assert not self.closed and self.pending is None,'Drain previous frame before scheduling another'
        assert raw.ndim==2 and crop.shape==(478,478) and raw.dtype==crop.dtype==np.uint8
        self.pending=self.pool.submit(write_images,Path(base),raw.copy(),crop.copy())

    def finish(self):
        if self.pending is None:return None
        future=self.pending
        try:return future.result()
        finally:self.pending=None

    def close(self):
        try:return self.finish()
        finally:self.pool.shutdown(wait=True,cancel_futures=False);self.closed=True

    def __enter__(self):return self

    def __exit__(self,kind,error,traceback):
        # Never cancel a captured frame, including after a signal-guard exception.
        self.close()
