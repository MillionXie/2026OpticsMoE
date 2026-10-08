import unittest
from unittest.mock import patch
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read,STAGES
from LightGenV2.tasks.t03_saliency.lab_supervise import completed_prefix

class ResumeTests(unittest.TestCase):
    def test_retry_transient_read(self):
        with patch.object(Path,'read_text',side_effect=[PermissionError(),'{"ok":true}']) as f,patch('time.sleep'):
            self.assertEqual(read('state.json'),{'ok':True})
            self.assertEqual(f.call_count,2)

    def test_persistent_error_is_not_hidden(self):
        with patch.object(Path,'read_text',side_effect=PermissionError()) as f,patch('time.sleep'):
            with self.assertRaises(PermissionError):read('state.json')
            self.assertEqual(f.call_count,61)

    def test_only_contiguous_stages(self):
        self.assertEqual(completed_prefix(STAGES,list(STAGES[:5])),5)
        with self.assertRaises(ValueError):completed_prefix(STAGES,[STAGES[1]])

if __name__=='__main__':unittest.main()
