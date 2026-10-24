import unittest
import numpy as np
from .diagnose_measured_expert import survey_indices
from .probe_expert_exposure import candidate_hardware


class ExpertSurveyTest(unittest.TestCase):
    def test_fixed_cross_class_TRAIN_identities(self):
        labels=np.repeat(np.arange(10),100)
        selected=survey_indices(labels)
        self.assertEqual(selected,[0,50,150,250,350,450,550,650,750,850,950])
        self.assertEqual(set(labels[selected]),set(range(10)))

    def test_reject_incomplete_or_unbalanced_TRAIN(self):
        with self.assertRaises(ValueError):survey_indices(np.zeros(1000,dtype=int))
        with self.assertRaises(ValueError):survey_indices(np.arange(10))

    def test_new_exposure_identity_preserves_geometry_and_LUT(self):
        from .hardware_bridge import hardware_identity
        old=dict(exposure_us=1999.8,coordinates='coordinates',coordinates_SHA256='geometry',
                 phase_lut='phase',phase_LUT_SHA256='phase_sha',response_lut='response',
                 response_lut_SHA256='response_sha',camera_device_info={'Exposure':1999.8})
        changed=candidate_hardware(old,3000,'parent_sha')
        self.assertEqual(old['exposure_us'],1999.8)
        for key in ('coordinates','coordinates_SHA256','phase_lut','phase_LUT_SHA256','response_lut','response_lut_SHA256'):
            self.assertEqual(old[key],changed[key])
        self.assertNotEqual(hardware_identity(old),hardware_identity(changed))
        self.assertNotIn('camera_device_info',changed)
        with self.assertRaises(ValueError):candidate_hardware(old,5000,'parent')


if __name__=='__main__':unittest.main()
