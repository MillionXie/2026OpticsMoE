import ast
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
import importlib.util

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_closedloop_resume_20261021.py'
spec=importlib.util.spec_from_file_location('resume_contract',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RecoveryContract(unittest.TestCase):
    def test_no_algorithm_or_settings_patch(self):
        text=SOURCE.read_text();ast.parse(text)
        self.assertNotIn('m.Bench=',text);self.assertNotIn('.settings(',text)
        self.assertIn('loop.acquire(CaptureAPI(m,bench)',text)
        self.assertIn('loop.refine(',text);self.assertIn('next_layer_started=False',text)
        self.assertIn('background_subtracted=False',text)
        self.assertIn('same_connection_as_formal_capture=True',text)

    def test_zero_only_recoverable_archive(self):
        class M:
            @staticmethod
            def read(p):return json.loads(p.read_text())
            @staticmethod
            def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);M.LOCK=root/'lock';folder=root/'L6_round1';capture=folder/'capture';capture.mkdir(parents=True)
            (capture/'failure.json').write_text(json.dumps(dict(completed_receipts=0,error="RuntimeError('DVP Start: -1001')")))
            (capture/'valid_raw.png').write_bytes(b'protected')
            with self.assertRaises(AssertionError):module.archive_zero_frame_failure(M,folder)
            self.assertTrue((capture/'valid_raw.png').exists())
            (capture/'valid_raw.png').unlink()
            result=module.archive_zero_frame_failure(M,folder)
            self.assertFalse(capture.exists());self.assertFalse(result['deleted'])
            self.assertEqual(result['valid_frames_moved'],0)
            self.assertTrue((folder/'capture_failed_start_20261021/failure.json').exists())


if __name__=='__main__':unittest.main()
