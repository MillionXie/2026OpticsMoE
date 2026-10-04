import tempfile
from pathlib import Path
import unittest

from maintenance.git_safety.plan_main_checkout import blob_sha, classify_bytes

class ClassificationTests(unittest.TestCase):
    def classify(self,data,expected):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample'
            path.write_bytes(data)
            result=classify_bytes(path,blob_sha(expected))
            self.assertEqual(path.read_bytes(),data)
            return result

    def test_identical(self):
        self.assertEqual(self.classify(b'unchanged\n',b'unchanged\n'),'byte_identical_to_main')

    def test_checkout_newlines(self):
        self.assertEqual(self.classify(b'line\r\n',b'line\n'),'crlf_only_difference_from_main')

    def test_real_edit_not_normalized_away(self):
        self.assertEqual(self.classify(b'keep user edit\r\n',b'new main\n'),'different_from_main_preserve_and_review')

    def test_binary_difference_preserved(self):
        self.assertEqual(self.classify(b'\0\xff\x01',b'\0\xff\x02'),'different_from_main_preserve_and_review')

if __name__=='__main__':unittest.main()
