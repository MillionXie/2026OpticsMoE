import unittest
import numpy as np
from .hardware_bridge import native_image,crop_native,guard,signal,acquisition_order,TRAIN_control_indices,repeat_policy,admission_checks,signal_guard_policy,stage_SNR_threshold
import copy,json
from pathlib import Path


class TestHardware(unittest.TestCase):
    def test_C_SNR8_is_exact_case_only_and_recorded_not_global_default(self):
        authorization=json.loads(Path(__file__).with_name('config_LG_C_SNR8_20261025.json').read_text(encoding='utf-8'))
        base=dict(mode='pointwise_dark_p99',authorization='Original fixed2ms permission',minimum_signal_counts=64,minimum_SNR=10,exposure_us=2000,single_frame=True,network_background_subtraction=False)
        package=dict(architecture='moe',stage='C',checkpoint_sha256=authorization['checkpoint_sha256'],hardware_sha256=authorization['hardware_sha256'],C_SNR8_authorization=authorization,signal_guard_policy=base,TRAIN_stage_signal_calibration=True)
        for stage in ('router','first','final'):
            self.assertEqual(stage_SNR_threshold(package,dict(exposure_us=1999.8),stage),8)
            self.assertEqual(stage_SNR_threshold({},dict(exposure_us=2000),stage),10)
        for key,value in (('architecture','d2nn'),('stage','D'),('checkpoint_sha256','other'),('hardware_sha256','other'),('TRAIN_stage_signal_calibration',False)):
            with self.assertRaises(ValueError):stage_SNR_threshold(dict(package,**{key:value}),dict(exposure_us=2000),'first')
        changed=copy.deepcopy(package);changed['C_SNR8_authorization']['minimum_signal_counts']=63
        with self.assertRaises(ValueError):stage_SNR_threshold(changed,dict(exposure_us=2000),'first')
        with self.assertRaises(ValueError):stage_SNR_threshold(package,dict(exposure_us=4000),'first')

    def test_C_SNR8_changes_only_signal_ratio_not_other_guards(self):
        row=dict(saturation_fraction=0,readout_p99_counts=1000,readout_p10_counts=500,fixed_dark_difference_p99_counts=229)
        noise=dict(read_noise_upper_counts=26.77)
        with self.assertRaises(RuntimeError):guard(row,noise,policy='pointwise_dark_p99')
        guard(row,noise,policy='pointwise_dark_p99',minimum_snr=8)
        for key,value in (('saturation_fraction',.151),('fixed_dark_difference_p99_counts',210),('readout_p10_counts',750)):
            with self.assertRaises(RuntimeError):guard(dict(row,**{key:value}),noise,policy='pointwise_dark_p99',minimum_snr=8)
        values=dict(snr=8.56,delta99=229,correlation=.79,repeat_drift=.0331,marker_change=180,noise=26.77,saturation=0,mode='record_only')
        self.assertFalse(admission_checks(**values)[0]);self.assertTrue(admission_checks(**values,minimum_snr=8)[0])
        for key,value in (('delta99',63),('repeat_drift',.051),('marker_change',70),('saturation',.151)):
            self.assertFalse(admission_checks(**dict(values,**{key:value}),minimum_snr=8)[0])

    def test_pointwise_policy_requires_fixed_guard_authority(self):
        p=dict(mode='pointwise_dark_p99',authorization='User approved 2026-10-25',minimum_signal_counts=64,minimum_SNR=10,exposure_us=2000,single_frame=True,network_background_subtraction=False)
        self.assertEqual(signal_guard_policy(),'legacy');self.assertEqual(signal_guard_policy(p,1999.8),'pointwise_dark_p99')
        for bad in (dict(p,authorization=''),dict(p,minimum_SNR=9),dict(p,minimum_signal_counts=63),dict(p,network_background_subtraction=True)):
            with self.assertRaises(ValueError):signal_guard_policy(bad,2000)
    def test_spatial_dark_quantile_is_not_difference_of_quantiles(self):
        dark=np.zeros((100,100),np.float32);dark[:,:20]=1200/65535
        value=dark.copy();value[:,20:40]=440/65535;original=value.copy();noise=dict(read_noise_upper_counts=28,dark_global_p99_counts=1200)
        with self.assertRaises(RuntimeError):guard(signal(value),noise)
        row=signal(value,dark=dark);guard(row,noise,policy='pointwise_dark_p99');np.testing.assert_array_equal(value,original)
        self.assertAlmostEqual(row['fixed_dark_difference_p99_counts'],440,places=3)
        with self.assertRaises(RuntimeError):guard(dict(row,fixed_dark_difference_p99_counts=279),noise,policy='pointwise_dark_p99')
        with self.assertRaises(RuntimeError):guard(dict(row,saturation_fraction=.151),noise,policy='pointwise_dark_p99')
        with self.assertRaises(RuntimeError):guard(dict(row,readout_p10_counts=1199),noise,policy='pointwise_dark_p99')
        with self.assertRaises(ValueError):guard(signal(value),noise,policy='pointwise_dark_p99')
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
