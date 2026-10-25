import json
import os
from pathlib import Path
import tempfile
import unittest

from .build_lab_package import sha
from .lab_capture import sdk_owner, validate_handoff


class LabCaptureTests(unittest.TestCase):
    def test_idle_failure_is_not_completed_handoff(self):
        for row in ({}, {'SDK_released': True}, {'T21_capture_complete':False,'SDK_released':True,'next_task':'T20'}):
            with self.assertRaises(ValueError):
                validate_handoff(row)

    def test_complete_expected_D_only(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'report.json'
            r=dict(passed=True,pilot=False,SDK_released=True,full_physical_prefix=True,
                checkpoint_sha256='D_pt',stages=['router','first','final'],records=5900,frames=17700)
            p.write_text(json.dumps(r))
            handoff=dict(T21_capture_complete=True,SDK_released=True,next_task='T20',report=str(p),
                report_sha256=sha(p),checkpoint_sha256='D_pt')
            validate_handoff(handoff)
            for key, value in (('frames',17699),('pilot',True),('records',1000),('SDK_released',False)):
                p.write_text(json.dumps(dict(r,**{key:value})))
                with self.assertRaises(ValueError):
                    validate_handoff(dict(handoff,report_sha256=sha(p)))

    def test_exclusive_lock_preserves_other_owner(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'lock.json'
            with sdk_owner(p, Path(name)):
                self.assertEqual(json.loads(p.read_text())['pid'],os.getpid())
                before=p.read_bytes()
                with self.assertRaises(FileExistsError):
                    with sdk_owner(p, Path(name)):
                        self.fail('Second SDK owner admitted')
                self.assertEqual(before,p.read_bytes())
            self.assertFalse(p.exists())

    def test_exception_releases_only_owned_lock(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'lock.json'
            with self.assertRaisesRegex(RuntimeError,'training'):
                with sdk_owner(p, Path(name)):
                    raise RuntimeError('training')
            self.assertFalse(p.exists())


if __name__=='__main__':
    unittest.main()
