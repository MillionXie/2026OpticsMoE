import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import caltech_selected_extend_finalize as f
import audit_caltech_selected_extend as audit


class Finalize(unittest.TestCase):
    def test_changed_test_claim_exclusive_and_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);key=f.e.KEYS[0]
            f.claim(out,key,'lock-digest',False)
            self.assertIn('before first test materialization',f.e.read(out/key/'test_started.json')['policy'])
            with self.assertRaises(FileExistsError):f.claim(out,key,'lock-digest',False)

    def test_reuse_claim_is_distinct_and_also_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);key=f.e.KEYS[0];f.claim(out,key,'lock-digest',True)
            self.assertEqual(f.e.read(out/key/'test_started.json')['policy'],'Reuse saved parent test; no inference')
            with self.assertRaises(FileExistsError):f.claim(out,key,'lock-digest',True)

    def test_all_eight_endpoint_barrier_rejects_missing_last_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            f.save(out/'status.json',dict(state='tests_complete_awaiting_cpu_audit',failures=[]))
            f.save(out/'current_processes.json',dict(active=[],pending=[]));f.save(out/'suite_started.json',dict(pid=0))
            for key in f.e.KEYS[:-1]:
                f.save(out/key/'test_started.json',dict(pid=0))
                f.save(out/key/'result.json',dict(test_evaluations=1,metrics=dict(train={},val={},test={}),test_used_for_selection=False,selection_split='val'))
            with self.assertRaises(FileNotFoundError):audit.endpoint_barrier(out)

    def test_all_eight_endpoint_barrier_accepts_completed_reuse_and_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            f.save(out/'status.json',dict(state='tests_complete_awaiting_cpu_audit',failures=[]))
            f.save(out/'current_processes.json',dict(active=[],pending=[]));f.save(out/'suite_started.json',dict(pid=0))
            for i,key in enumerate(f.e.KEYS):
                f.save(out/key/'test_started.json',dict(pid=0))
                f.save(out/key/'result.json',dict(test_evaluations=i%2,metrics=dict(train={},val={},test={}),test_used_for_selection=False,selection_split='val'))
            audit.endpoint_barrier(out)


if __name__=='__main__':unittest.main()
