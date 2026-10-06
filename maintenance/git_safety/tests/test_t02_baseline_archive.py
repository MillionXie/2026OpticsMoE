import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from maintenance.storage.check_t02_baseline_archive import verify


class BaselineArchiveTests(unittest.TestCase):
    def fixture(self, root, incorrect_mean=False, duplicate=False, wrong_reported_head=False):
        head = b'opaque checkpoint bytes, never deserialized'
        digest = lambda b: hashlib.sha256(b).hexdigest()
        report = dict(test_samples=2, timing_samples=2, explicit_warmup_forwards=50,
                      first_test_sample_included=False, checkpoint_sha256='wrong' if wrong_reported_head else digest(head),
                      latency_cuda_ms=dict(mean=9 if incorrect_mean else 1.5, median=1.5, std=.5, p05=1.05, p95=1.95, min=1, max=2),
                      latency_host_ms=dict(mean=2.5, median=2.5, std=.5, p05=2.05, p95=2.95, min=2, max=3),
                      power=dict(idle_samples=1, active_samples=1, idle_mean_w=40, active_mean_w=50,
                                 active_peak_w=50, measured_active_energy_j_per_sample=.075))
        payloads = dict(report=json.dumps(report).encode(), head=head, command=b'fixed command, not executed',
                        timing=b'sample_index,sample_id,cuda_ms,host_ms\n0,a,1,2\n1,b,2,3\n',
                        power=b'phase,watts\nidle,40\nactive:0,50\n')
        path = root / 'original.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, raw in payloads.items():
                member = tarfile.TarInfo(name); member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
            if duplicate:
                raw = payloads['report']; member = tarfile.TarInfo('report'); member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        official = root / 'split.csv'
        official.write_bytes(b'sample_id,split\na,test\nb,test\ntrain,train\n')
        spec = dict(archive_sha256=digest(path.read_bytes()), report_fields=dict(test_samples=2, timing_samples=2, explicit_warmup_forwards=50),
                    members={name: dict(member=name, bytes=len(raw), sha256=digest(raw)) for name, raw in payloads.items()},
                    official_manifest_sha256=digest(official.read_bytes()))
        return path, official, spec

    def test_raw_binding_and_optional_test_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            path, official, spec = self.fixture(Path(folder))
            original = path.read_bytes()
            result = verify(path, spec, official)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['timing_samples'], 2)
            self.assertFalse(result['archive_extracted'])
            self.assertFalse(result['model_loaded'])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ['original.tar.gz', 'split.csv'])

    def test_wrong_summary_is_not_treated_as_a_new_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, spec = self.fixture(Path(folder), incorrect_mean=True)
            self.assertIn('Raw timing summary mismatch: latency_cuda_ms.mean', verify(path, spec)['errors'])

    def test_reported_head_must_match_preserved_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, spec = self.fixture(Path(folder), wrong_reported_head=True)
            self.assertIn('Reported head SHA does not bind preserved head', verify(path, spec)['errors'])

    def test_duplicates_and_changed_archive_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, spec = self.fixture(Path(folder), duplicate=True)
            self.assertEqual(verify(path, spec)['errors'], ['Duplicate archive member names'])
            spec['archive_sha256'] = 'different'
            self.assertEqual(verify(path, spec)['errors'], ['Archive SHA mismatch'])


if __name__ == '__main__':
    unittest.main()
