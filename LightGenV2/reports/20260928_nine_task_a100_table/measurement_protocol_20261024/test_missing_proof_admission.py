"""No model/GPU: independent missing-only systems must not block each other."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_t12_cost as queue


class MissingProofAdmission(unittest.TestCase):
    def test_ours_plan_never_reads_qwen_proof(self):
        with patch.object(queue,'predecessor_file_ready',return_value=True) as ready:
            self.assertTrue(queue.required_proofs_ready(queue.point_plan(['ours_repeat01','ours_repeat02','ours_repeat03']),'ours','qwen'))
            ready.assert_called_once_with('ours','ours')

    def test_qwen_plan_never_reads_ours_proof(self):
        with patch.object(queue,'predecessor_file_ready',return_value=False) as ready:
            self.assertFalse(queue.required_proofs_ready(queue.point_plan(['qwen_repeat01']),'ours','qwen'))
            ready.assert_called_once_with('qwen','qwen')

    def test_both_plan_requires_both(self):
        with patch.object(queue,'predecessor_file_ready',side_effect=[True,False]) as ready:
            self.assertFalse(queue.required_proofs_ready(queue.point_plan(['ours_repeat01','qwen_repeat01']),'ours','qwen'))
            self.assertEqual(ready.call_count,2)

    def test_invalid_point_remains_rejected(self):
        with self.assertRaises(ValueError):queue.point_plan(['ours_repeat04'])


if __name__=='__main__':unittest.main()
