"""No SDK/Torch: byte identity, buffer ownership and bounded failure semantics."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dvp_persistence',HERE/'lab_caltech20_dvp_persistence_20261019.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.raw=np.random.default_rng(17).integers(0,80,(512,640),np.uint8);self.crop=self.raw[:478,:478].copy()

    def test_identical_serial_PNG_bytes_and_SHA(self):
        Image.fromarray(self.raw).save(self.root/'serial_raw.png',compress_level=1);Image.fromarray(self.crop).save(self.root/'serial.png',compress_level=1)
        with m.FrameWriter() as writer:
            writer.submit(self.root/'async',self.raw,self.crop);row=writer.finish()
        self.assertEqual((self.root/'serial_raw.png').read_bytes(),(self.root/'async_raw.png').read_bytes())
        self.assertEqual((self.root/'serial.png').read_bytes(),(self.root/'async.png').read_bytes())
        self.assertEqual(row['raw_SHA256'],hashlib.sha256((self.root/'serial_raw.png').read_bytes()).hexdigest())

    def test_SDK_buffer_reuse_cannot_mutate_snapshot(self):
        expected_raw=self.raw.copy();expected_crop=self.crop.copy()
        with m.FrameWriter() as writer:
            writer.submit(self.root/'owned',self.raw,self.crop);self.raw.fill(255);self.crop.fill(255);writer.finish()
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'owned_raw.png')),expected_raw))
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'owned.png')),expected_crop))

    def test_queue_cannot_grow(self):
        with m.FrameWriter() as writer:
            writer.submit(self.root/'first',self.raw,self.crop)
            with self.assertRaises(AssertionError):writer.submit(self.root/'second',self.raw,self.crop)
            writer.finish();writer.submit(self.root/'second',self.raw,self.crop);writer.finish()

    def test_signal_exception_still_saves_bad_frame(self):
        with self.assertRaisesRegex(RuntimeError,'signal'):
            with m.FrameWriter() as writer:
                writer.submit(self.root/'bad',self.raw,self.crop);raise RuntimeError('signal')
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'bad_raw.png')),self.raw))

    def test_IO_failure_is_not_suppressed(self):
        with self.assertRaises(FileNotFoundError):
            with m.FrameWriter() as writer:writer.submit(self.root/'missing/frame',self.raw,self.crop);writer.finish()

    def test_existing_capture_is_never_overwritten(self):
        path=self.root/'old_raw.png';path.write_bytes(b'protected')
        with self.assertRaises(AssertionError):
            with m.FrameWriter() as writer:writer.submit(self.root/'old',self.raw,self.crop);writer.finish()
        self.assertEqual(path.read_bytes(),b'protected')


if __name__=='__main__':unittest.main()
