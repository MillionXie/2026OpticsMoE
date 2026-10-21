from pathlib import Path
import ast
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_closedloop_audit_20261021.py'


class RecoveryAudit(unittest.TestCase):
    def test_old_failure_and_new_exact_connection_evidence(self):
        text=SOURCE.read_text();ast.parse(text)
        for value in ("returned['returncode']==0", "m.read(original_return)['returncode']==1",
            "['completed_receipts']==0",'diagnostic_raw_crop_BMP_frame_ids_verified=23',
            "['frame_id']>last",'repeat_admission(', 'USB_recovery_evidence=recovery_proof'):
            self.assertIn(value,text)
        self.assertNotIn('.Bench(',text);self.assertNotIn('.settings(',text)


if __name__=='__main__':unittest.main()
