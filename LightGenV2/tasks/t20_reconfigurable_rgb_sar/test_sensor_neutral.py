import unittest
import torch
from .sensor_neutral import encode
from .unified_native import encode_single


class SensorNeutralTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_RGB_contract_unchanged(self):
        x=torch.randint(1,255,(2,56,56,3),dtype=torch.uint8);f=torch.rand(2,128)
        torch.testing.assert_close(encode(x,f),encode_single(x,torch.zeros(2,dtype=torch.int64),f),rtol=0,atol=0)
    def test_all_channels_preserved_same_layout_without_sensor_argument(self):
        x=torch.zeros(1,56,56,3,dtype=torch.uint8);x[:,:,:,2]=128;f=torch.ones(1,128)
        a=encode(x,f);self.assertGreater(float(a[:,112:,:112].sum()),0.);self.assertEqual(float(a[:,112:,112:].sum()),0.);torch.testing.assert_close(a.square().sum((1,2)),torch.ones(1))


if __name__=='__main__':unittest.main()
