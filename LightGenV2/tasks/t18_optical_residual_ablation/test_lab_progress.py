"""Exercise Windows progress publication fallback without importing torch/SDKs."""
import ast
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


class ProgressTest(unittest.TestCase):
    def test_busy_progress_preserves_receipt(self):
        source=Path(__file__).with_name('lab_caltech10_20261015.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='save')
        ns={'Path':Path,'json':json,'time':time}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'save_test','exec'),ns)
        original=Path.replace
        def busy(path,target):
            if Path(target).name=='progress.json':raise PermissionError('Windows reader')
            return original(path,target)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as directory:
            target=Path(directory)/'progress.json';target.write_text('{"old":true}')
            with patch.object(Path,'replace',busy),patch.object(time,'sleep'):
                ns['save'](target,{'frames':21})
            self.assertEqual(json.loads(target.read_text()),{'old':True})
            snapshots=list(Path(directory).glob('progress_*.json'))
            self.assertEqual(len(snapshots),1)
            self.assertEqual(json.loads(snapshots[0].read_text()),{'frames':21})


if __name__=='__main__':unittest.main()
