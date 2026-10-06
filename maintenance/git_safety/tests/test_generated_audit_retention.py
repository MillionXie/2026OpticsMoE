"""Check retained generated audit identities without deleting or moving files."""
import hashlib,json,subprocess,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
class RetentionTests(unittest.TestCase):
    def test_original_audits_recoverable_and_ignored(self):
        record=json.loads((ROOT/'maintenance/storage/T06_GENERATED_AUDIT_RETENTION_20261006.json').read_text())
        self.assertEqual(len(record['files']),7)
        for row in record['files']:
            path=record['prefix']+row['path']
            data=subprocess.check_output(['git','-C',str(ROOT),'show',record['original_git_commit']+':'+path])
            self.assertEqual(hashlib.sha256(data).hexdigest(),row['git_blob_sha256'])
            result=subprocess.run(['git','-C',str(ROOT),'check-ignore','--no-index',path],capture_output=True)
            self.assertEqual(result.returncode,0)
            local=ROOT/path
            if local.is_file():
                self.assertEqual(local.read_bytes().replace(b'\r\n',b'\n'),data)
    def test_source_and_weights_do_not_enter_generated_audit_scope(self):
        record=json.loads((ROOT/'maintenance/storage/T06_GENERATED_AUDIT_RETENTION_20261006.json').read_text())
        for row in record['files']:
            self.assertIn(Path(row['path']).suffix,{'.csv','.json'})
        self.assertFalse(record['timing_data_models_CCD_removed'])
if __name__=='__main__':unittest.main()
