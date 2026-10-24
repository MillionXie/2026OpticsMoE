import unittest
import torch
from .transport_TRAIN_readout import response_map,normalized_transport,response_perturbation


class TRAINTransportTests(unittest.TestCase):
    def test_training_response_preserves_mean_one_and_generalizes_known_linear(self):
        torch.manual_seed(3);x=1+.1*torch.randn(100,12,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        target=x.roll(1,1);m=response_map(x[:80],target[:80],.000001)
        predicted=normalized_transport(x[80:],m)
        self.assertLess(float((predicted-target[80:]).square().mean()),1e-8)
        self.assertTrue(torch.allclose(predicted.mean(1),torch.ones(20,dtype=torch.float64)))

    def test_wrong_paired_count_rejected(self):
        with self.assertRaises(ValueError):response_map(torch.ones(4,3),torch.ones(5,3),.1)

    def test_response_noise_uses_only_requested_TRAIN_class(self):
        residual=torch.cat((torch.zeros(100,8,dtype=torch.float64),torch.ones(100,8,dtype=torch.float64)))
        labels=torch.cat((torch.zeros(100,dtype=torch.long),torch.ones(100,dtype=torch.long)))
        result=response_perturbation(residual,labels,torch.tensor([0,1,0,1]))
        self.assertEqual(float(result[[0,2]].abs().max()),0.)
        self.assertTrue(torch.isfinite(result).all());self.assertGreater(float(result[[1,3]].abs().max()),0.)


if __name__=='__main__':unittest.main()
