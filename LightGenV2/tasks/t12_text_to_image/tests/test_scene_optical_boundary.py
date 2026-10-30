import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.scene_optical_boundary import physical_field,checked_ccd,STAGES,attach


class Boundary(unittest.TestCase):
    def test_complex_and_negative_input_phase_is_not_discarded(self):
        field=torch.tensor([[[0.,-.4,.2j,-.3j,.7]]]);phase=torch.linspace(0,5,5).reshape(1,1,5)
        modulation=torch.exp(1j*phase);a,p=physical_field(field,modulation)
        torch.testing.assert_close(a*torch.exp(1j*p),field*modulation,rtol=1e-5,atol=1e-6)
        self.assertEqual(float(a[0,0,0]),0)
        with self.assertRaises(ValueError):physical_field(field*10,modulation)
    def test_missing_or_malformed_CCD_never_falls_back_to_simulation(self):
        field=torch.zeros(1,2,2)
        for value in [None,torch.zeros(1,3,3),torch.full((1,2,2),float('nan')),torch.full((1,2,2),-1.)]:
            with self.assertRaises(ValueError):checked_ccd(value,field,2)
        torch.testing.assert_close(checked_ccd(torch.ones(1,2,2),field,2),torch.ones(1,2,2))
    def test_real_Qwen_dependency_order_and_single_pair_only(self):
        self.assertEqual(STAGES,('vision_router','vision_expert','vision_global','language_router','language_expert','language_global'))
        with self.assertRaises(ValueError):attach({},lambda *args:None)


if __name__=='__main__':unittest.main()
