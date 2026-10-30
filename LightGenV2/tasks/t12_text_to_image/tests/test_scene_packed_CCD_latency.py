import unittest
from ..scene_packed_CCD_latency import packed_pixels,cpu_reference_pixels


class PackedPayloadTest(unittest.TestCase):
    def test_tensor_byte_equivalence_including_rounding_boundaries(self):
        import torch
        import numpy as np
        a=torch.cat((torch.linspace(0,1,2049),torch.arange(255).float().add(.5).div(255)))
        p=torch.linspace(0,2*torch.pi,len(a));actual=packed_pixels(a,p)
        self.assertEqual(actual.dtype,torch.uint8);self.assertTrue(actual.is_contiguous())
        np.testing.assert_array_equal(actual.numpy(),cpu_reference_pixels(a,p))
        self.assertEqual(int(actual[0,0]),0);self.assertEqual(int(actual[0,2048]),255)

    def test_random_physical_field_shapes_and_no_peak_normalization(self):
        import torch
        import numpy as np
        torch.manual_seed(1042);a=torch.rand(1,478,478)*.23;p=torch.rand_like(a)*2*torch.pi
        actual=packed_pixels(a,p);self.assertEqual(tuple(actual.shape),(2,1,478,478))
        self.assertLess(int(actual[0].max()),60)
        np.testing.assert_array_equal(actual.numpy(),cpu_reference_pixels(a,p))

    def test_changed_native_dtype_is_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError,'float32'):packed_pixels(torch.ones(2,dtype=torch.bfloat16),torch.zeros(2))


if __name__=='__main__':unittest.main()
