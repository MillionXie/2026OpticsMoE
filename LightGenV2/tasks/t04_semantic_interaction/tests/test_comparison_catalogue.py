"""Numbering and full/split PDF lookup must stay one-to-one."""
import unittest
import tempfile
from pathlib import Path
from LightGenV2.tasks.t04_semantic_interaction.comparison_catalogue import (
    OPERATIONS, plan_pages, ensure_artifact_ignore, GENERATED_IGNORE,
)


class CataloguePlanTest(unittest.TestCase):
    def test_generated_outputs_get_scoped_ignore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_artifact_ignore(root)
            self.assertEqual((root / '.gitignore').read_text(), GENERATED_IGNORE)
            ensure_artifact_ignore(root)  # Idempotent; does not touch assets.
            self.assertNotIn('*.json', GENERATED_IGNORE)
            self.assertNotIn('*.py', GENERATED_IGNORE)
            self.assertNotIn('README', '\n'.join(line for line in GENERATED_IGNORE.splitlines()
                                               if not line.startswith('#')))

    def test_custom_ignore_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.gitignore').write_text('/custom_asset.png\n')
            with self.assertRaises(ValueError):
                ensure_artifact_ignore(root)
            self.assertEqual((root / '.gitignore').read_text(), '/custom_asset.png\n')

    def test_complete_page_plan(self):
        pairs = []
        for split, count in (('test', 250), ('train', 1250)):
            for operation in OPERATIONS:
                for i in range(count):
                    pairs.append({'number': len(pairs)+1, 'split': split, 'operation': operation,
                                  'sample_id': f'{split}_{OPERATIONS.index(operation)+4*i:06d}'})
        sections, pages = plan_pages(pairs)
        self.assertEqual(pages, {'all': 1513, 'test': 257, 'train': 1257})
        self.assertEqual(len(sections), 8)
        self.assertEqual(len({r['folder'] for r in pairs}), 6000)
        self.assertEqual(pairs[0]['master_pdf_page'], 3)
        self.assertEqual(pairs[999]['master_pdf_page'], 257)
        self.assertEqual(pairs[1000]['master_pdf_page'], 259)
        self.assertEqual(pairs[-1]['master_pdf_page'], 1513)
        self.assertEqual(pairs[-1]['split_pdf_page'], 1257)
        self.assertTrue(all(r['folder'].startswith('01_test/') for r in pairs[:1000]))
        self.assertTrue(all(r['folder'].startswith('02_train/') for r in pairs[1000:]))


if __name__ == '__main__':
    unittest.main()
