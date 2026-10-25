import unittest
import numpy as np
from .hardware_bridge import native_image,crop_native,guard,signal,acquisition_order,TRAIN_control_indices,repeat_policy,admission_checks


class TestHardware(unittest.TestCase):
    def test_repeat_exception_is_explicit_and_fixed_2ms(self):
        self.assertEqual(repeat_policy(),'enforce')
        p=dict(mode='record_only',authorization='Human 2026-10-25 approval',exposure_us=2000,single_frame=True)
        self.assertEqual(repeat_policy(p,2000),'record_only')
        self.assertEqual(repeat_policy(p,1999.8),'record_only')  # Camera's quantized 2 ms readback.
        for bad,exposure in ((dict(p,authorization=''),2000),(dict(p,single_frame=False),2000),
                             (dict(p,mode='disable_all'),2000),(p,4000),(dict(p,extra=True),2000)):
            with self.assertRaises(ValueError):repeat_policy(bad,exposure)

    def test_only_repeat_metric_can_be_diagnostic(self):
        values=dict(snr=16.8,delta99=100,correlation=.927,repeat_drift=.0304,
                    marker_change=100,noise=10,saturation=0)
        self.assertFalse(admission_checks(**values)[0])
        passed,checks=admission_checks(**values,mode='record_only')
        self.assertTrue(passed);self.assertFalse(checks['repeat_correlation'])
        for field,bad in (('snr',9),('delta99',63),('repeat_drift',.051),
                          ('marker_change',29),('saturation',.151),('correlation',float('nan'))):
            changed=dict(values,**{field:bad})
            self.assertFalse(admission_checks(**changed,mode='record_only')[0],field)

    def test_fixed_order_preserves_record_identity(self):
        rows=[dict(split='train' if i<2 else 'test',task='eurosat',local_index=i%2,label=i%2) for i in range(4)]
        ordered=acquisition_order(rows,[3,0,2,1])
        self.assertEqual(ordered,[rows[3],rows[0],rows[2],rows[1]])
        self.assertEqual(TRAIN_control_indices(ordered,[dict(task='eurosat',local_index=0),dict(task='eurosat',local_index=1)]),[1,3])
        self.assertIs(acquisition_order(rows,None),rows)
        for bad in ([0,0,2,3],[0,1,2],[-1,0,1,2],[False,1,2,3]):
            with self.assertRaises(ValueError):acquisition_order(rows,bad)
        with self.assertRaises(ValueError):TRAIN_control_indices(ordered,[dict(task='eurosat',local_index=0,split='test'),dict(task='eurosat',local_index=1)])

    def test_native_panels(self):
        x=np.zeros((1016,1016),np.uint8);x[17,29]=87
        amplitude=native_image(x);phase=native_image(x,True)
        self.assertEqual(amplitude.shape,(1080,1920));self.assertEqual(phase.shape,(1200,1920))
        np.testing.assert_array_equal(amplitude[32:1048,452:1468],x)
        np.testing.assert_array_equal(255-phase[92:1108,452:1468],np.flipud(np.fliplr(x)))
        self.assertEqual(int(amplitude[:32].max()),0)
        with self.assertRaises(ValueError):native_image(np.zeros((478,478),np.uint8))

    def test_full_precision_crop(self):
        raw=np.full((2048,2048),12345,np.uint16)
        result=crop_native(raw,np.eye(3,dtype=np.float64))
        self.assertEqual(result.dtype,np.float32);self.assertEqual(result.shape,(1016,1016))
        np.testing.assert_allclose(result,12345/65535,atol=1e-7,rtol=0)
        self.assertNotAlmostEqual(float(result[0,0])*255,round(float(result[0,0])*255),places=3)

    def test_signal_admission_not_8bit_threshold_bypass(self):
        noise=dict(read_noise_upper_counts=50,dark_router_ROI_p99_counts=920,dark_global_p99_counts=520)
        good=dict(p99_gray=3.2,saturation_fraction=0,readout_p99_counts=3300,readout_p10_counts=450)
        guard(good,noise,router=True)
        with self.assertRaises(RuntimeError):guard(dict(good,readout_p99_counts=1000),noise,router=True)
        with self.assertRaises(RuntimeError):guard(dict(good,saturation_fraction=.16),noise,router=True)
        with self.assertRaises(RuntimeError):guard(good)


if __name__=='__main__':unittest.main()
