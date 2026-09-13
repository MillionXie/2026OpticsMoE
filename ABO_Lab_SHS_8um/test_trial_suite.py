import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import trial_suite

class SuiteTests(unittest.TestCase):
    def test_pending_dashboard_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'results/run01';out.mkdir(parents=True)
            current=root/'reports/00_current'
            state=dict(status='running',active='01',trials=[dict(name='01',status='running',exposure_us=400,wait_ms=250)])
            with patch.object(trial_suite,'ROOT',root),patch.object(trial_suite,'CURRENT',current):
                trial_suite.publish(out,state)
            self.assertIn('400',(current/'00_READ_ME.md').read_text(encoding='utf-8'))
            self.assertTrue((current/'99_history_index.html').is_file())
            self.assertIn('run01',(current/'99_history_index.html').read_text(encoding='utf-8'))
            self.assertIn('99_history_index.html',(current/'01_summary.html').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
