import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run


class StageTests(unittest.TestCase):
    def invoke(self,*,stage='vision_router',completed=(),returncodes=(0,0)):
        args=SimpleNamespace(stage=stage,session='pilot02',device='cuda',config='custom lab.json')
        samples=[{'id':'title_000','kind':'title'},{'id':'image_0','kind':'image'},
                 {'id':'image_1','kind':'image'}]
        records=lambda root,s:{stage:{}} if s['id'] in completed else {}
        with patch.object(run,'load_session',return_value=(Path('session'),{'samples':samples})),\
             patch.object(run,'measured',side_effect=records),\
             patch.object(run,'config',return_value=({},Path('custom lab.json').resolve())),\
             patch.object(run.subprocess,'run',side_effect=[SimpleNamespace(returncode=v) for v in returncodes]) as sub:
            try: run.stage(args,{})
            except SystemExit as ex: return sub.call_args_list,ex.code
            return sub.call_args_list,0

    def test_preparation_then_capture_in_separate_processes(self):
        calls,code=self.invoke()
        self.assertEqual(code,0); self.assertEqual(len(calls),2)
        first,second=[c.args[0] for c in calls]
        self.assertEqual(first[:3],[sys.executable,str(run.ROOT/'run.py'),'prepare'])
        self.assertEqual(second[:3],[sys.executable,str(run.ROOT/'run.py'),'capture'])
        self.assertEqual(first[-2:],['--device','cuda'])
        self.assertIn(str(Path('custom lab.json').resolve()),first)
        self.assertEqual(calls[1].kwargs,{}) # inherits stdin, no auto-confirm input

    def test_prepare_failure_never_opens_capture(self):
        calls,code=self.invoke(returncodes=(7,))
        self.assertEqual(code,7); self.assertEqual(len(calls),1)

    def test_capture_failure_propagates(self):
        calls,code=self.invoke(returncodes=(0,3))
        self.assertEqual(code,3); self.assertEqual(len(calls),2)

    def test_completed_vision_ignores_uncaptured_titles(self):
        calls,code=self.invoke(completed=('image_0','image_1'))
        self.assertEqual((len(calls),code),(0,0))

    def test_language_still_needs_titles(self):
        calls,code=self.invoke(stage='language_router',completed=('image_0','image_1'))
        self.assertEqual((len(calls),code),(2,0))

    def test_completed_language_skips_hardware(self):
        calls,code=self.invoke(stage='language_router',completed=('image_0','image_1','title_000'))
        self.assertEqual((len(calls),code),(0,0))


if __name__=='__main__': unittest.main()
