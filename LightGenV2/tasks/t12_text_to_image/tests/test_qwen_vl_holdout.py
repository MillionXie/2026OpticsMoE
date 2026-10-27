import json
import tempfile
import unittest
from pathlib import Path

from ..qwen_vl_holdout import check_pairs, exposure_inventory, reject_exposure


class HoldoutTest(unittest.TestCase):
    def test_both_roles_and_development_excluded(self):
        prior = [{'split': 'val', 'source': {'product_id': 'a', 'family_id': 'f', 'rgba_sha256': 'x'},
                  'target': {'product_id': 'b', 'family_id': 'g', 'rgba_sha256': 'y'}}]
        for candidate in [{'product_id': 'b', 'family_id': 'z', 'rgba_sha256': 'z'},
                          {'product_id': 'c', 'family_id': 'g', 'rgba_sha256': 'z'},
                          {'product_id': 'c', 'family_id': 'z', 'rgba_sha256': 'x'}]:
            with self.assertRaises(ValueError):
                reject_exposure([candidate], prior)
        self.assertEqual(reject_exposure([{'product_id': 'c', 'family_id': 'z', 'rgba_sha256': 'z'}], prior)['product_id'], 2)

    def test_follows_early_resume_and_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['current', 'parent', 'smoke']:
                (root/name).mkdir()
                (root/name/'annotated_pairs.json').write_text('[]')
            (root/'current/execution.json').write_text(json.dumps({'argv': ['--parent', 'parent']}))
            (root/'parent/execution.json').write_text(json.dumps({'argv': ['--resume', 'smoke/best_checkpoint.pt']}))
            (root/'smoke/execution.json').write_text(json.dumps({'argv': []}))
            _, audit = exposure_inventory(root/'current', root)
            self.assertEqual(len(audit['ancestor_runs']), 3)
            (root/'smoke/annotated_pairs.json').unlink()
            with self.assertRaises(ValueError):
                exposure_inventory(root/'current', root)

    def test_balanced_test_and_no_training_rows(self):
        rows = [{'sample_id': str(i), 'split': 'test', 'category': cat, 'mode': mode}
                for i, (cat, mode) in enumerate((cat, mode) for cat in ['lamp', 'pillow']
                    for mode in ['background', 'object', 'joint'] for _ in range(40))]
        self.assertEqual(len(check_pairs(rows)), 6)
        rows[0]['split'] = 'train'
        with self.assertRaises(ValueError):
            check_pairs(rows)


if __name__ == '__main__':
    unittest.main()
