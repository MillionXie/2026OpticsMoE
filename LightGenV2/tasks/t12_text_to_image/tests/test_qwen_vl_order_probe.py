import unittest
import torch

from LightGenV2.tasks.t12_text_to_image.qwen_vl_order_probe import relative_change,probe_ids


class OrderProbeTests(unittest.TestCase):
    def test_exact_identity_and_relative_scaling(self):
        x=torch.randn(64,16)
        self.assertEqual(relative_change(x,x)['max_abs'],0)
        self.assertGreater(relative_change(x,x+1)['relative_RMS'],0)
        with self.assertRaises(ValueError):relative_change(x,x[:1])

    def test_only_first_two_distinct_train_products(self):
        rows=[]
        for c in ('lamp','pillow'):
            for split in ('val','train'):
                for product in ('a','a','b','c'):
                    rows.append({'split':split,'category':c,'source':{'product_id':product}})
        selected=[rows[i] for i in probe_ids(rows)]
        self.assertEqual(len(selected),4)
        self.assertTrue(all(r['split']=='train' for r in selected))
        self.assertEqual([r['source']['product_id'] for r in selected],['a','b','a','b'])
