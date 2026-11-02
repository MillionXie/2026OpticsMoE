import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from ..scene_turbo_diagnostic import verify,UPSTREAM


class TurboAssetTests(unittest.TestCase):
    def test_changed_base_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);(root/'weight').write_bytes(b'changed')
            report=root/'assets.json'
            report.write_text(json.dumps(dict(complete=True,upstream_commit=UPSTREAM,
                base_files={'weight':dict(sha256='0'*64)},edge_checkpoint=dict(sha256='1'*64))))
            a=SimpleNamespace(assets=report,upstream=root,base=root,checkpoint=root/'edge')
            with patch('subprocess.check_output',side_effect=[UPSTREAM+'\n','']):
                with self.assertRaisesRegex(ValueError,'base asset drift'):verify(a)

    def test_modified_upstream_rejected_before_asset_access(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);report=root/'assets.json'
            report.write_text(json.dumps(dict(complete=True,upstream_commit=UPSTREAM)))
            a=SimpleNamespace(assets=report,upstream=root)
            with patch('subprocess.check_output',side_effect=[UPSTREAM+'\n',' M src/model.py\n']):
                with self.assertRaisesRegex(ValueError,'Do not modify'):verify(a)


if __name__=='__main__':unittest.main()
