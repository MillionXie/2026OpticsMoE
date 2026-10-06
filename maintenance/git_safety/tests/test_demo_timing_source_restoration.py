"""Exact historical timing source identities; no inference or measurement."""
import hashlib,json,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
class DemoTimingSourceRestoration(unittest.TestCase):
    def test_all_fifteen_original_hashes_preserved(self):
        d=json.loads((ROOT/'maintenance/storage/DEMO_TIMING_SOURCE_RESTORATION_20261007.json').read_text(encoding='utf8'))
        self.assertEqual(len(d['files']),15)
        for r in d['files']:
            p=(ROOT/r['path']).resolve()
            self.assertTrue(p.is_relative_to(ROOT))
            self.assertFalse(p.is_symlink())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['sha256'])

    def test_git_preserves_original_bytes_only_for_named_snapshots(self):
        d=json.loads((ROOT/'maintenance/storage/DEMO_TIMING_SOURCE_RESTORATION_20261007.json').read_text(encoding='utf8'))
        for r in d['files']:
            result=subprocess.check_output(['git','-C',str(ROOT),'check-attr','text','--',r['path']],text=True)
            self.assertTrue(result.strip().endswith(': unset'))
        attrs=(ROOT/'LightGenV2/.gitattributes').read_text(encoding='utf8')
        self.assertNotIn('reports/** -text',attrs)
if __name__=='__main__':unittest.main()
