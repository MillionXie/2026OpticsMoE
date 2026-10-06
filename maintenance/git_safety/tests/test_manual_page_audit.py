from pathlib import Path
import tempfile
import unittest
from maintenance.git_safety.audit_manual_pages import inspect_pages


class Doc:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __len__(self): return 3
    def __getitem__(self, index):
        class Page:
            def get_text(self): return str(index)
        return Page()


class ManualTests(unittest.TestCase):
    def test_one_based_page_identity_and_no_output_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manual.pdf'; path.write_bytes(b'fixture')
            result = inspect_pages(path, [1, 3], opener=lambda p: Doc())
            self.assertEqual(result['pages'], [{'page': 1, 'text': '0'}, {'page': 3, 'text': '2'}])
            self.assertFalse(result['measured_timing'])
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_invalid_page_contract(self):
        with self.assertRaises(ValueError): inspect_pages('unused.pdf', [0])
        with self.assertRaises(ValueError): inspect_pages('unused.pdf', [1, 1])

    def test_out_of_range_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manual.pdf'; path.write_bytes(b'fixture')
            with self.assertRaises(ValueError): inspect_pages(path, [4], opener=lambda p: Doc())
