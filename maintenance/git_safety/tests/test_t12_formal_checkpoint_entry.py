import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from LightGenV2.tasks.t12_text_to_image import verify_formal_checkpoint as entry


class CheckpointEntryTests(unittest.TestCase):
    def test_adapted_requires_explicit_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture.pt'; path.write_bytes(b'adapted')
            model = SimpleNamespace(kind='small', parameters=lambda: iter([
                SimpleNamespace(numel=lambda: entry.PARAMETERS, device=SimpleNamespace(type='cpu'))]))
            with patch.object(entry, 'ADAPTED_PIN', hashlib.sha256(b'adapted').hexdigest()):
                with self.assertRaises(ValueError):
                    entry.inspect_checkpoint(path, loader=lambda p: self.fail('Wrong variant loaded'))
                result = entry.inspect_checkpoint(path, loader=lambda p: model, variant='decoder-adapted')
                self.assertEqual(result['variant'], 'decoder-adapted')

    def test_identity_rejected_before_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture.pt'; path.write_bytes(b'fixture')
            with self.assertRaises(ValueError):
                entry.inspect_checkpoint(path, loader=lambda p: self.fail('Loader must not run'))

    def test_cpu_contract_and_no_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture.pt'; path.write_bytes(b'fixture')
            model = SimpleNamespace(kind='small', parameters=lambda: iter([
                SimpleNamespace(numel=lambda: entry.PARAMETERS, device=SimpleNamespace(type='cpu'))]))
            with patch.object(entry, 'PIN', hashlib.sha256(b'fixture').hexdigest()):
                result = entry.inspect_checkpoint(path, loader=lambda p: model)
            self.assertTrue(result['read_only'])
            self.assertEqual(path.read_bytes(), b'fixture')
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_wrong_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture.pt'; path.write_bytes(b'fixture')
            model = SimpleNamespace(kind='small', parameters=lambda: iter([]))
            with patch.object(entry, 'PIN', hashlib.sha256(b'fixture').hexdigest()), self.assertRaises(ValueError):
                entry.inspect_checkpoint(path, loader=lambda p: model)


if __name__ == '__main__':
    unittest.main()
