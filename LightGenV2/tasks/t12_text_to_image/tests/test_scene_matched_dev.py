"""Check that a comparison cannot silently select TEST, a prompt or a seed."""
import unittest
from LightGenV2.tasks.t12_text_to_image.appearance_scene_protocol import conditions
from LightGenV2.tasks.t12_text_to_image.scene_matched_dev import select_requests


class MatchedScopeTest(unittest.TestCase):
    def rows(self):
        result = []
        for i in range(16):
            category = 'lamp' if i < 8 else 'pillow'
            for j in range(2):
                result.append({**conditions(category)[j], 'split': 'dev',
                    'reference_id': str(i), 'reference': str(i)+'.png', 'seed': 1042+i})
        return result

    def test_sealed_test_is_not_selected(self):
        rows = self.rows()
        self.assertEqual(select_requests(rows+[dict(rows[0], split='test')]), rows)

    def test_missing_dev_cannot_be_filled_from_test(self):
        rows = self.rows()
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError): select_requests(rows)

    def test_prompt_and_input_identity_are_checked(self):
        rows = self.rows()
        rows[0]['prompt'] += ' Change the shade color.'
        with self.assertRaises(ValueError): select_requests(rows)
        rows = self.rows()
        rows[-1] = rows[-2].copy()
        with self.assertRaises(ValueError): select_requests(rows)


if __name__ == '__main__': unittest.main()
