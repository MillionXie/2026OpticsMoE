import unittest
from .probe_B_expert_photons import candidate,fixed_receipt_indices


class BPhotonProbeTests(unittest.TestCase):
    def test_only_predeclared_exposures(self):
        with self.assertRaises(ValueError):candidate(dict(exposure_us=4000),16000,'x')
        with self.assertRaises(ValueError):candidate(dict(exposure_us=2000),6000,'x')

    def test_hardware_fields_preserved(self):
        parent=dict(exposure_us=4000,coordinates='C',phase_lut='P',response_lut='R',orientation='identity',phase_transform='hv_inverse',camera_device_info='old')
        value=candidate(parent,8000,'parent')
        for name in ('coordinates','phase_lut','response_lut','orientation','phase_transform'):self.assertEqual(value[name],parent[name])
        self.assertEqual(parent['exposure_us'],4000);self.assertFalse(value['network_background_subtraction'])

    def test_fixed_pilot_source_rows_only(self):
        self.assertEqual(fixed_receipt_indices(48),(0,12,24,47))
        with self.assertRaises(ValueError):fixed_receipt_indices(2000)


if __name__=='__main__':unittest.main()
