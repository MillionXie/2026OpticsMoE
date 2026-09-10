import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import offload as o


class TestOffload(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(o.normalized('assets\\test/a.png'),'assets/test/a.png')
        for p in ['../x','/tmp/x','C:/x','a/../../b']:
            with self.assertRaises(ValueError): o.normalized(p)

    def test_stage_dependencies(self):
        self.assertEqual(o.required({'kind':'title'}),('language_router','language_expert'))
        self.assertEqual(len(o.required({'kind':'image'})),5)

    def test_unpack_checks_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);p=t/'input.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('a.txt',b'valid');z.writestr('SNAPSHOT.json',json.dumps({'files':{'a.txt':o.file_hash(b'valid')}}))
            o.unpack_snapshot(p,t/'ok');self.assertEqual((t/'ok/a.txt').read_bytes(),b'valid')
            with zipfile.ZipFile(t/'bad.zip','w') as z:
                z.writestr('a.txt',b'bad');z.writestr('SNAPSHOT.json',json.dumps({'files':{'a.txt':o.file_hash(b'valid')}}))
            with self.assertRaises(ValueError):o.unpack_snapshot(t/'bad.zip',t/'bad')

    def test_export_stops_before_zip_if_missing(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);r=root/'sessions/pilot';r.mkdir(parents=True)
            c={'amplitude_slm':{'gray_lut_file':None}}
            o.write(r/'session.json',{'hardware_identity':o.hardware_identity(c),'checkpoint_sha256':o.CHECKPOINT_SHA,
                'samples':[{'id':'title_000','kind':'title'}]})
            with patch.object(o,'ROOT',root),patch.object(o,'session_path',return_value=r),patch.object(o,'config',return_value=(c,None)):
                with self.assertRaisesRegex(ValueError,'Missing upstream'):o.export_snapshot('pilot',root/'out.zip')
            self.assertFalse((root/'out.zip').exists())


if __name__=='__main__':unittest.main()
