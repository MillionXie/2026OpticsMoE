import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import evaluate_offload as e
from common import STAGES, write, sha, CHECKPOINT_SHA


class OffloadTests(unittest.TestCase):
    def test_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x'; p.write_bytes(b'a')
            files = {'x': sha(p)}
            e.verify_files(d, files)
            p.write_bytes(b'b')
            with self.assertRaises(ValueError): e.verify_files(d, files)

    def test_all_real_stages(self):
        self.assertEqual((*e.required({'kind': 'image'}), 'language_global'), tuple(STAGES))
        self.assertEqual((*e.required({'kind': 'title'}), 'language_global'), tuple(STAGES[3:]))

    def test_changed_session_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            write(r/'session.json', {'samples': []})
            write(r/'base.json', {'manifest': {'session': 'pilot02', 'checkpoint_sha256': CHECKPOINT_SHA,
                  'files': {'session.json': 'wrong'}}})
            with patch.object(e, 'config', return_value=({}, None)), patch.object(e, 'session_path', return_value=r):
                with self.assertRaisesRegex(ValueError, 'Session changed'):
                    e.export_delta('pilot02', r/'base.json', r/'out.zip')
            self.assertFalse((r/'out.zip').exists())


if __name__ == '__main__': unittest.main()
