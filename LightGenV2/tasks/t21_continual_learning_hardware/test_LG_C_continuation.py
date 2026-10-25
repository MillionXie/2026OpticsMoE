import unittest,json
from pathlib import Path
from .continue_LG_C_optics import validate,native_retention_loss
import torch


class CContinuationTests(unittest.TestCase):
    def test_finite_budget_and_original_memory(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_optical_continuation_20261025.json').read_text());validate(c)
        for key,value in [('epochs',20),('memory_records_per_old_task',1000),('TEST_gradients',True)]:
            with self.assertRaises(ValueError):validate(dict(c,**{key:value}))
        native=json.loads(Path(__file__).with_name('config_LG_C_native_B_route_20261025.json').read_text());validate(native)
        self.assertEqual(native['epochs'],1);self.assertEqual(native['memory_records_per_old_task'],300)

    def test_native_route_preserves_teacher_and_has_gradient(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_native_B_route_20261025.json').read_text())
        teacher=torch.zeros(2,16);teacher[:,:8]=1/8
        logits=torch.zeros(2,10,requires_grad=True);raw=torch.zeros(2,16,requires_grad=True)
        loss=native_retention_loss(dict(logits=logits,route_power=raw.softmax(1)),torch.zeros(2,10),teacher,c);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(raw.grad[:,8:].sum()),0)
        self.assertTrue(torch.equal(teacher[:,:8],torch.full((2,8),1/8)))


if __name__=='__main__':unittest.main()
