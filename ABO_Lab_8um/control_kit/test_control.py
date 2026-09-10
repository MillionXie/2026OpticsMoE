import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import control


class FakeDevice:
    def __init__(self): self.closed = False; self.calls = []
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def device_info(self): return {'test_fake': True}
    def preload_files(self, p): self.calls.append(('preload', p))
    def display_file(self, p): self.calls.append(('display', p))
    def capture(self, p): np.save(p, np.arange(12, dtype=np.uint16).reshape(3,4))


class Tests(unittest.TestCase):
    def test_raw_configuration(self):
        c, _ = control.load_config()
        self.assertEqual(c['camera']['saved_frame_resize_mode'], 'none')
        self.assertIsNone(c['camera']['saved_frame_size_wh'])

    def test_no_postprocessing_allowed(self):
        c, _ = control.load_config(); c['camera']['saved_frame_size_wh'] = [478,478]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.json'; p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): control.load_config(p)

    def test_patterns(self):
        c,_=control.load_config()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'patterns'; control.patterns(c,p)
            with Image.open(p/'checker64.bmp') as im:
                self.assertEqual(im.size,(1920,1080)); self.assertEqual(im.mode,'L')
                a=np.asarray(im).copy(); self.assertEqual(int(a[0,0]),0); self.assertEqual(int(a[0,64]),255)
                im.close()  # Pillow BMP memory mapping can outlive its file context on Windows.
            with self.assertRaises(FileExistsError): control.patterns(c,p)

    def fake_controller(self):
        hw=control.Controller.__new__(control.Controller)
        hw.config,_=control.load_config(); hw.slm=FakeDevice(); hw.camera=FakeDevice()
        hw.stack=contextlib.ExitStack(); hw.last_display=None
        return hw

    def test_sequence_capture_pixels_metadata_and_close(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); bmp=p/'a.bmp'; Image.new('L',(1920,1080),128).save(bmp)
            hw=self.fake_controller()
            with patch('control.time.sleep') as sleep:
                with hw:
                    hw.display(bmp); raw=hw.capture(p/'out',frames=2)
                sleep.assert_called_once_with(.2)
            self.assertTrue(hw.slm.closed); self.assertTrue(hw.camera.closed)
            self.assertEqual([x[0] for x in hw.slm.calls],['preload','display'])
            np.testing.assert_array_equal(raw,np.arange(12,dtype=np.uint16).reshape(3,4))
            with Image.open(p/'out/0001.tif') as im: np.testing.assert_array_equal(raw,np.asarray(im))
            report=json.loads((p/'out/capture.json').read_text())
            self.assertEqual(report['postprocessing'],'none'); self.assertEqual(len(report['frames']),2)
            self.assertEqual(report['display']['sha256'],control.digest(bmp))
            with self.assertRaises(FileExistsError): hw.capture(p/'out')

    def test_reject_rgb(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'rgb.bmp'; Image.new('RGB',(1920,1080)).save(p)
            with self.assertRaises(ValueError): self.fake_controller().display(p)

    def test_failure_keeps_record_and_files(self):
        with tempfile.TemporaryDirectory() as t:
            hw=self.fake_controller()
            def fail(p):
                np.save(p,np.array([1],dtype=np.uint8)); raise RuntimeError('mock failure')
            hw.camera.capture=fail
            with self.assertRaises(RuntimeError): hw.capture(Path(t)/'out')
            self.assertTrue((Path(t)/'out/0000.npy').exists())
            report=json.loads((Path(t)/'out/capture.json').read_text())
            self.assertTrue(report['interrupted_or_failed'])

    def test_driver_import_without_hardware(self):
        d=control.drivers()
        self.assertTrue(hasattr(d,'HoloeyeSLM')); self.assertTrue(hasattr(d,'DvpSubprocessCamera'))


if __name__=='__main__': unittest.main()
