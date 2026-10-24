import unittest
import torch
from .transport_TRAIN_readout import response_map,normalized_transport


class TRAINTransportTests(unittest.TestCase):
    def test_training_response_preserves_mean_one_and_generalizes_known_linear(self):
        torch.manual_seed(3);x=1+.1*torch.randn(100,12,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        target=x.roll(1,1);m=response_map(x[:80],target[:80],.000001)
        predicted=normalized_transport(x[80:],m)
        self.assertLess(float((predicted-target[80:]).square().mean()),1e-8)
        self.assertTrue(torch.allclose(predicted.mean(1),torch.ones(20,dtype=torch.float64)))

    def test_wrong_paired_count_rejected(self):
        with self.assertRaises(ValueError):response_map(torch.ones(4,3),torch.ones(5,3),.1)


if __name__=='__main__':unittest.main()
