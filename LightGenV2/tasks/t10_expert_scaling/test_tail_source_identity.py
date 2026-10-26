"""CPU boundary checks for continuing a locked 90-epoch run."""
import json
from pathlib import Path
import tempfile
import unittest

from .expert_tail_schedule import audit_source, sha


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / 'best_checkpoint.pt').write_bytes(b'best90')
        (self.path / 'last_checkpoint.pt').write_bytes(b'last90-with-optimizer')
        arguments = dict(experts=49, top_k=49, seed=17, layers=4, epochs=90)
        (self.path / 'metadata.json').write_text(json.dumps(dict(arguments=arguments, data_sha256='data')))
        (self.path / 'result.json').write_text(json.dumps(dict(checkpoint_sha256=sha(self.path / 'best_checkpoint.pt'))))
        (self.path / 'history.json').write_text(json.dumps([dict(epoch=90)]))
        self.job = dict(source=str(self.path), experts=49, top_k=49, seed=17,
                        source_epochs=90, best_sha256=sha(self.path / 'best_checkpoint.pt'),
                        last_sha256=sha(self.path / 'last_checkpoint.pt'))

    def test_locked90_accepted_but_default60_rejects_it(self):
        self.assertEqual(audit_source(self.job)['source_epochs'], 90)
        job = dict(self.job)
        del job['source_epochs']
        with self.assertRaises(ValueError):
            audit_source(job)

    def test_chained_source_requires_unchanged_last(self):
        job = dict(self.job)
        del job['last_sha256']
        with self.assertRaises(ValueError):
            audit_source(job)
        (self.path / 'last_checkpoint.pt').write_bytes(b'replaced-model')
        with self.assertRaises(ValueError):
            audit_source(self.job)

    def test_metadata_cannot_hide_incomplete_history(self):
        (self.path / 'history.json').write_text(json.dumps([dict(epoch=89)]))
        with self.assertRaises(ValueError):
            audit_source(self.job)


if __name__ == '__main__':
    unittest.main()
