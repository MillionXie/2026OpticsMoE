import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from maintenance.git_safety.format_adopted_checkout import plan,apply


class CheckoutFormattingTests(unittest.TestCase):
    def test_only_canonical_git_bytes_and_explicit_eligibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            subprocess.run(['git','-C',folder,'init','-q'],check=True)
            subprocess.run(['git','-C',folder,'config','core.autocrlf','false'],check=True)
            canonical=b'value = 1\n'
            (root/'source.py').write_bytes(canonical)
            subprocess.run(['git','-C',folder,'add','source.py'],check=True)
            subprocess.run(['git','-C',folder,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','source'],check=True)
            pin=subprocess.check_output(['git','-C',folder,'rev-parse','HEAD']).decode().strip()
            raw=b'value = 1\r\n';(root/'source.py').write_bytes(raw)
            row={'path':'source.py','classification':'line_endings_only',
                 'raw_sha256':hashlib.sha256(raw).hexdigest(),'expected_sha256':hashlib.sha256(canonical).hexdigest()}
            with self.assertRaises(ValueError):plan(root,'git',pin,[row],set())
            rows=plan(root,'git',pin,[row],{'source.py'})
            self.assertEqual(apply(root,'git',pin,rows)['repaired_files'],1)
            self.assertEqual((root/'source.py').read_bytes(),canonical)
            self.assertFalse(subprocess.check_output(['git','-C',folder,'diff','--name-only']).strip())


if __name__=='__main__':unittest.main()
