"""CPU-only rank72 bench projection contracts; no real SDK or devices."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from LightGenV2.tasks.t07_abo_image_retrieval.hardware import bench, geometry


class FakeCamera:
    fresh_calls = 0
    def get(self, name):
        return {'ExposureTime':'400', 'Gain':'Gain_X4', 'AcquisitionFrameRate':'100'}[name]
    def fresh(self):
        self.fresh_calls += 1


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'machine.json'
        self.config.write_text(json.dumps({'camera':{},'amplitude_slm':{}}))
        self.phase = self.root / 'phase.bmp'
        self.amplitude = self.root / 'amplitude.bmp'
        Image.fromarray(np.zeros((1200,1920),np.uint8)).save(self.phase)
        Image.fromarray(np.zeros((1080,1920),np.uint8)).save(self.amplitude)
        self.events = []
        events = self.events
        class Controller:
            def __init__(inner, config, *, config_base):
                inner.c, inner.base = config, config_base
                inner.camera, inner.slm = FakeCamera(), object()
                inner.raw = np.full((1080,1920),20,np.uint8)
            def __enter__(inner):
                events.append('controller_enter')
                return inner
            def __exit__(inner,*args): events.append('controller_exit')
            def capture(inner, path):
                events.append('amplitude_capture')
                return inner.raw, {'frame_id':1,'timestamp_ns':2,'settle_drained_frames':53}
        class Phase:
            def __init__(inner,*args,**kwargs):
                inner.show_calls=0
                inner.kwargs=kwargs
            def __enter__(inner): events.append('phase_enter'); return inner
            def __exit__(inner,*args): events.append('phase_exit')
            def show(inner,path): inner.show_calls+=1; events.append('phase_show'); return {}
        self.patches = [patch.object(bench,'Controller',Controller),patch.object(bench,'PhaseHDMI',Phase),
                        patch.object(bench,'snapshot',lambda camera: {})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def construct(self):
        return bench.SHSBench(self.root,400,240,{},machine_config=self.config,
                              phase_sdk=self.root,phase_lut=self.phase)

    def test_explicit_config_is_not_modified(self):
        before = self.config.read_bytes()
        value = self.construct()
        self.assertEqual(self.config.read_bytes(),before)
        self.assertEqual(value.controller.base,self.root)
        self.assertEqual(value.controller.c['camera'],{'exposure_us':400,'gain':'Gain_X4'})
        self.assertEqual(value.controller.c['settle_delay_ms'],240)
        self.assertEqual(value.phase.kwargs,{'settle_s':.8,'pixel_format':'rgba'})
        self.assertEqual(self.events,[])

    def test_entry_capture_and_release_order(self):
        value = self.construct()
        with value:
            image, _ = value.capture('vision_router',self.phase,[self.amplitude],['sample'],'flip_v',save=False)
            self.assertEqual(image.shape,(1,478,478))
            self.assertTrue(np.all(image == 20))
            self.assertEqual(value.rows[0]['p99'],20)
            self.assertTrue(value.rows[0]['no_photometric_normalization'])
            self.assertEqual(value.rows[0]['settle_drained_frames'],53)
        self.assertEqual(self.events,['phase_enter','controller_enter','phase_show','amplitude_capture','controller_exit','phase_exit'])

    def test_amplitude_sdk_override_preserves_machine_file(self):
        before=self.config.read_bytes()
        sdk=self.root/'vendor'
        sdk.mkdir()
        value=bench.SHSBench(self.root,400,240,{},machine_config=self.config,
                             phase_sdk=self.root,phase_lut=self.phase,amplitude_sdk=sdk)
        self.assertEqual(value.controller.c['amplitude_slm']['sdk_path'],str(sdk.resolve()))
        self.assertEqual(self.config.read_bytes(),before)
        self.assertEqual(self.events,[])

    def test_missing_amplitude_sdk_rejected_before_device_construction(self):
        with patch.object(bench,'Controller') as controller, patch.object(bench,'PhaseHDMI') as phase:
            with self.assertRaises(FileNotFoundError):
                bench.SHSBench(self.root,400,240,{},machine_config=self.config,
                              phase_sdk=self.root,phase_lut=self.phase,amplitude_sdk=self.root/'missing')
            controller.assert_not_called()
            phase.assert_not_called()

    def test_same_phase_not_reshown(self):
        with self.construct() as value:
            for _ in range(2):
                value.capture('vision_router',self.phase,[self.amplitude],['sample'],'flip_v',save=False)
            self.assertEqual(value.phase.show_calls,1)
            self.assertEqual(value.camera.fresh_calls,1)

    def test_wrong_native_frame_shape_rejected(self):
        with self.construct() as value:
            value.controller.raw = np.zeros((478,478),np.uint8)
            with self.assertRaisesRegex(RuntimeError,'Unexpected SHS frame shape'):
                value.capture('vision_router',self.phase,[self.amplitude],['sample'],'flip_v',save=False)

    def test_saturated_frame_rejected(self):
        with self.construct() as value:
            value.controller.raw[:] = 255
            with self.assertRaisesRegex(RuntimeError,'saturated'):
                value.capture('vision_router',self.phase,[self.amplitude],['sample'],'flip_v',save=False)

    def test_complete_receipt_pair_written_to_fixture_only(self):
        with self.construct() as value:
            value.capture('vision_router',self.phase,[self.amplitude],['sample'],'flip_v')
        folder = self.root / 'ccd/vision_router'
        self.assertTrue((folder/'sample.png').is_file())
        receipt = json.loads((folder/'sample.json').read_text())
        self.assertEqual(receipt['amplitude_sha256'],geometry.sha(self.amplitude))
        self.assertEqual(receipt['phase_sha256'],geometry.sha(self.phase))
        self.assertEqual(receipt['wait_ms'],240)

    def test_native_canvas_zero_padding_and_shape(self):
        active = np.full((478,478),127,np.uint8)
        amplitude = geometry.active_to_native(active)
        phase = geometry.active_to_native(active,'phase')
        self.assertEqual(amplitude.shape,(1080,1920))
        self.assertEqual(phase.shape,(1200,1920))
        self.assertEqual(np.count_nonzero(amplitude),1016**2)
        self.assertEqual(np.count_nonzero(phase),1016**2)

    def test_entry_failure_still_closes_both_devices(self):
        value = self.construct()
        with patch.object(value.camera if hasattr(value,'camera') else value.controller.camera,
                          'get', side_effect=RuntimeError('camera read failed')):
            with self.assertRaisesRegex(RuntimeError,'camera read failed'):
                value.__enter__()
        self.assertEqual(self.events,['phase_enter','controller_enter','controller_exit','phase_exit'])

    def test_close_failure_does_not_skip_phase_close(self):
        value = self.construct()
        with patch.object(type(value.controller),'__exit__',side_effect=RuntimeError('close failed')):
            with self.assertRaisesRegex(RuntimeError,'close failed'):
                value.__exit__(None,None,None)
        self.assertEqual(self.events,['phase_exit'])


if __name__ == '__main__': unittest.main()
