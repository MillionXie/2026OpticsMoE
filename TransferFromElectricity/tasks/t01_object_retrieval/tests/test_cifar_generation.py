import random
import unittest
import torch
from TransferFromElectricity.tasks.t01_object_retrieval.datasets import class_partition
from TransferFromElectricity.tasks.t01_object_retrieval.models.generator import StaticGenerator


class CifarGenerationTests(unittest.TestCase):
    def test_gallery_is_disjoint_and_subset_class_selection_is_predefined(self):
        labels = [i//50 for i in range(500)]
        train,gallery = class_partition(labels,3,42,5)
        self.assertEqual((len(train),len(gallery)),(45,5))
        self.assertFalse(set(train)&set(gallery))
        self.assertEqual((train,gallery),class_partition(labels,3,42,5))
        self.assertEqual(sorted(random.Random(42).sample(range(100),10)),[3,13,14,17,28,31,35,81,86,94])

    def test_independent_output_heads_receive_only_their_expert_gradient(self):
        model=StaticGenerator('small_hyper',size=32,patch=8,expert_specific_heads=True)
        # CPU Transformer no-grad fastpath differs at float32 roundoff scale.
        torch.testing.assert_close(model(),model.anchor,rtol=0,atol=1e-7)
        model()[0,0].square().mean().backward()
        gradient=model.decoder.decode[-1].weight.grad
        self.assertGreater(float(gradient[0].norm()),0.)
        self.assertEqual(float(gradient[1:].abs().max()),0.)
        state=model.compact_state()
        clone=StaticGenerator('small_hyper',size=32,patch=8,expert_specific_heads=True)
        clone.load_compact_state(state)
        torch.testing.assert_close(model(),clone())


if __name__=='__main__': unittest.main()
