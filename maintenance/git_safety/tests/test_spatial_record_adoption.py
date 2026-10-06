"""Preserve historical server reports; do not re-evaluate scientific results."""
import hashlib,json,subprocess,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PREFIX='experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/'
HASHES={'SPATIAL_OPTIMIZATION_RESULT.md':'f0e69352c2586e588ebd38e8bcba35e08518838c9241f74857cdc3e23034eefa',
        'spatial_optimization_result.json':'ac498341c08cd0037900305fdda7e96460a4c32e8cc6d397dc25f7d233313b2d'}


class SpatialRecordTests(unittest.TestCase):
    def test_server_record_bytes_preserved(self):
        for name,expected in HASHES.items():
            self.assertEqual(hashlib.sha256((ROOT/(PREFIX+name)).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),expected)

    def test_internal_delta_and_historical_identity(self):
        report=json.loads((ROOT/(PREFIX+'spatial_optimization_result.json')).read_text())
        self.assertEqual(report['date'],'2026-09-10')
        self.assertEqual(report['checkpoint_sha256'],'6b05961f87f92173586504d3a984f7f9a9b473adf6a2d692777b4b9869e720fe')
        self.assertEqual(report['normal_optical_electronic']['count'],558)
        for key in ('srcc','krcc','plcc','rmse','mae'):
            self.assertAlmostEqual(report['normal_optical_electronic'][key]-report['same_checkpoint_optics_bypassed'][key],report['on_minus_off'][key])

    def test_exact_staged_incoming_record_fast_forward_preserves_unrelated_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def git(*args,data=None):
                return subprocess.check_output(['git','-C',folder,*args],input=data,stderr=subprocess.PIPE)
            git('init','-q')
            (root/'report.txt').write_text('old report\n')
            (root/'historical.txt').write_text('historical audit\n')
            git('add','--','report.txt','historical.txt')
            git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','old')
            old=git('rev-parse','HEAD').decode().strip()
            (root/'report.txt').write_text('actual server report\n')
            git('add','--','report.txt')
            tree=git('write-tree').decode().strip()
            pin=git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit-tree',tree,'-p',old,data=b'adopt report\n').decode().strip()
            (root/'historical.txt').unlink()
            git('merge','--ff-only','--no-edit',pin)
            self.assertEqual(git('rev-parse','HEAD').decode().strip(),pin)
            self.assertEqual((root/'report.txt').read_text(),'actual server report\n')
            self.assertFalse((root/'historical.txt').exists())
            self.assertEqual(git('status','--porcelain','-uno').decode().strip(),'D historical.txt')


if __name__=='__main__':unittest.main()
