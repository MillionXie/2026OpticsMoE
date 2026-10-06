import csv
import tempfile
import unittest
from pathlib import Path
from LightGenV2.tasks.t12_text_to_image.perceptual.assemble_three_task_scores import load, write_summary


class SummaryTests(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'input.csv').write_text('model,mode\nx,y\nx,y\n', encoding='utf8')
            with self.assertRaises(ValueError):
                load(root, 'input.csv')

    def test_archived_fifteen_rows_exact_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[3] / 'outputs/t12_perceptual_comparison_20260928'
        if not root.is_dir():
            self.skipTest('Private archived metric CSVs are not available on this machine')
        names = ('summary_lpips.csv', 'summary_dists.csv', 'summary_regions.csv', 'three_task_scores.csv')
        import hashlib
        before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new.csv'
            self.assertEqual(write_summary(root, output), 15)
            with output.open(newline='', encoding='utf8') as handle:
                actual = list(csv.DictReader(handle))
            with (root / 'three_task_scores.csv').open(newline='', encoding='utf8') as handle:
                expected = list(csv.DictReader(handle))
            self.assertEqual(actual, expected)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_summary(root, output)
            self.assertEqual(output.read_bytes(), saved)
        self.assertEqual(before, {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


if __name__ == '__main__':
    unittest.main()
