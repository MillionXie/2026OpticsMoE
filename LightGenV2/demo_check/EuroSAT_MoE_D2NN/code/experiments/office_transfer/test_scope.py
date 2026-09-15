import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from . import runtime,report

class ScopeContract(unittest.TestCase):
    def test_process_exits_during_proc_inspection(self):
        import run_office
        fake=MagicMock();fake.__truediv__.return_value=fake
        fake.read_text.return_value='123 (python) S 1'
        fake.read_bytes.return_value=b'python\0-m\0experiments.office_transfer\0--name\0M1\0'
        with patch.object(run_office,'Path',return_value=fake),patch.object(run_office.os,'readlink',side_effect=FileNotFoundError):
            self.assertFalse(run_office.preserved_child_alive(123,'M1'))

    def test_only_m1_and_sources_allowed(self):
        for stage in ('A','B_only','M1'):runtime.require_scope('pilot',stage)
        for stage in ('M2','D_A','D_B_only','D1'):
            with self.assertRaises(RuntimeError):runtime.require_scope('pilot',stage)
        with self.assertRaises(RuntimeError):runtime.require_scope('full')

    def test_report_needs_no_unapproved_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,a,b in [('A',.8,.6),('B_only',.3,.8),('M1',.79,.78)]:
                folder=root/'runs/pilot/seed42'/name;folder.mkdir(parents=True)
                (folder/'final.json').write_text(json.dumps(dict(validation_a={'accuracy':a},validation_b={'accuracy':b},total_train_samples=100,total_epoch_seconds=10,checkpoint='selected.pt')))
            with patch.object(report,'ROOT',root):result=report.summarize('pilot',('A','B_only','M1'))
            self.assertEqual(set(result['aggregate']),{'A','B_only','M1'})
            self.assertTrue(result['pilot_gate_passed'])
            self.assertIn('wait for user',result['next_action'])

if __name__=='__main__':unittest.main()
