import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import write

class AtomicWriteTests(unittest.TestCase):
 def test_transient_windows_reader_preserves_old_json(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json';write(p,{'n':1});original=Path.replace;calls=[]
   def replace(src,dst):
    calls.append(1)
    if len(calls)<3:
     self.assertEqual(json.loads(p.read_text()),{'n':1});raise PermissionError('reader holds file')
    return original(src,dst)
   with patch.object(Path,'replace',replace),patch('LightGenV2.tasks.t06_video_quality_assessment.lab_runtime.time.sleep'):
    write(p,{'n':2})
   self.assertEqual(json.loads(p.read_text()),{'n':2});self.assertEqual(len(calls),3)

 def test_permanent_permission_error_is_not_silenced(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json';write(p,{'n':1})
   with patch.object(Path,'replace',side_effect=PermissionError('locked')),patch('LightGenV2.tasks.t06_video_quality_assessment.lab_runtime.time.sleep'):
    with self.assertRaises(PermissionError):write(p,{'n':2})
   self.assertEqual(json.loads(p.read_text()),{'n':1})
