import unittest
from ..scene_microconditioning import descriptor


class MicroconditioningContract(unittest.TestCase):
    def test_historical_defaults_preserved(self):
        self.assertEqual(descriptor({},512)['inference_conditioning_side'],512)
        self.assertEqual(descriptor({'SDXL_conditioning_size':1024},512)['inference_conditioning_side'],1024)
        self.assertFalse(descriptor({},512)['explicit_DEV_override'])

    def test_explicit_DEV_override_records_training_and_inference(self):
        d=descriptor({'SDXL_conditioning_size':1024},512,512,expanded=True)
        self.assertEqual(d['training_conditioning_side'],1024);self.assertEqual(d['inference_conditioning_side'],512)
        self.assertEqual(d['output_pixel_side'],512);self.assertTrue(d['explicit_DEV_override'])
        for kwargs in [dict(scope='test',expanded=True),dict(expanded=False),dict(expanded=True,timing=True)]:
            with self.assertRaises(ValueError):descriptor({},512,512,**kwargs)
        with self.assertRaises(ValueError):descriptor({},512,256,expanded=True)


if __name__=='__main__':unittest.main()
