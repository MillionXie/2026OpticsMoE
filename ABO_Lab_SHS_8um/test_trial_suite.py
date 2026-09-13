import tempfile,unittest,json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import trial_suite

class SuiteTests(unittest.TestCase):
    def test_append_requires_completed_suite_and_same_count(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'results/run01';out.mkdir(parents=True)
            a=SimpleNamespace(out=out,trial=None,append_trial=(300,240),resume=False,publish_only=False,switch_count=80)
            with patch.object(trial_suite,'ROOT',root):
                for state in [dict(status='running',trials=[]),dict(status='complete',trials=[],switch_count=40)]:
                    (out/'suite.json').write_text(json.dumps(state),encoding='utf-8')
                    with self.assertRaises(ValueError):trial_suite.run(a)
                    self.assertEqual(json.loads((out/'suite.json').read_text()),state)

    def test_report_transient_lock_retry(self):
        with patch.object(trial_suite,'write',side_effect=[PermissionError('busy'),None]) as w, patch.object(trial_suite.time,'sleep'):
            trial_suite.write_report(Path('report.json'),{})
            self.assertEqual(w.call_count,2)

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
