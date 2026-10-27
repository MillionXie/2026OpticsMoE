import unittest

from LightGenV2.tasks.t12_text_to_image.qwen_vl_latency import summarize, validate_inputs


class TimingContractTests(unittest.TestCase):
    def test_target_or_hidden_input_rejected(self):
        row = {'sample_id': 'a', 'image': 'source.png', 'prompt': 'Change the background'}
        self.assertEqual(validate_inputs([row]), [row])
        for extra in ('target', 'mask', 'hidden', 'target_product_id'):
            with self.assertRaises(ValueError):
                validate_inputs([{**row, extra: 'forbidden'}])

    def test_duplicate_identity_rejected(self):
        row = {'sample_id': 'a', 'image': 'source.png', 'prompt': 'instruction'}
        with self.assertRaises(ValueError):
            validate_inputs([row, row])

    def test_summary_retains_first_cold_request(self):
        result = summarize([1000., 10., 10., 10.])
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 257.5)
        self.assertEqual(result['median'], 10.)
        self.assertEqual(result['max'], 1000.)
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == '__main__':
    unittest.main()
