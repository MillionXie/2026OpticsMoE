import unittest
from LightGenV2.tasks.t12_text_to_image.qwen_vl_identity_probe import select_identity_references


class IdentityProbeTests(unittest.TestCase):
    def test_fit_and_transfer_are_disjoint_products_from_original_train_only(self):
        rows=[]
        for split in ('val','train'):
            for category in ('lamp','pillow'):
                for product in range(10):
                    for view in range(2):rows.append({'split':split,'category':category,'source':{'product_id':f'{category}-{product}'}})
        fit,transfer=select_identity_references(rows)
        self.assertEqual(len(fit),12);self.assertEqual(len(transfer),6)
        self.assertTrue(all(rows[i]['split']=='train' for i in fit+transfer))
        self.assertFalse({rows[i]['source']['product_id'] for i in fit}&{rows[i]['source']['product_id'] for i in transfer})

    def test_insufficient_train_products_are_rejected(self):
        with self.assertRaises(ValueError):select_identity_references([])
