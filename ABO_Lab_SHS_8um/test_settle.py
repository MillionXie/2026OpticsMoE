import tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from PIL import Image
from slm_camera import Controller


class SettleTests(unittest.TestCase):
    def test_drain_during_wait_not_sleep(self):
        clock=[0.0]
        camera=Mock();camera.startup_warmup={}
        def grab():clock[0]+=.01
        camera.grab.side_effect=grab
        camera.fresh.return_value=(np.zeros((2,2),np.uint8),{})
        c={'amplitude_slm':{'expected_resolution_wh':[2,2]},'settle_delay_ms':50}
        hw=Controller(c);hw.camera=camera;hw.slm=Mock();hw.camera_settings={}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.bmp';Image.fromarray(np.zeros((2,2),np.uint8)).save(p)
            with patch('slm_camera.time.perf_counter',side_effect=lambda:clock[0]),patch('slm_camera.time.sleep',side_effect=AssertionError('No sleep-only settling')):
                _,meta=hw.capture(p)
        self.assertGreaterEqual(camera.grab.call_count,5)
        camera.fresh.assert_called_once()
        self.assertEqual(meta['settle_drained_frames'],camera.grab.call_count)


if __name__=='__main__':unittest.main()
