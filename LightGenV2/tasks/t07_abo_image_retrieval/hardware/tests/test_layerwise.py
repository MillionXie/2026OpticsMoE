"""CPU-only preflight and orchestration contracts; never open real devices."""
import ast
import builtins
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from LightGenV2.tasks.t07_abo_image_retrieval.hardware import layerwise as runner


class LayerwiseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        (self.data / 'image.png').touch()
        self.rows = [dict(sample_id=f'{i:016x}', split='train' if i < 1600 else 'query',
                          image_path='image.png', product_id=str(i // 12)) for i in range(2400)]
        self.args = SimpleNamespace(checkpoint=self.root/'weight.pt', checkpoint_sha256=runner.CHECKPOINT,
            processor=self.root/'processor', protocol=self.root/'protocol.json', data_root=self.data,
            geometry=self.root/'geometry.json', run_dir=self.root/'run', machine_config=self.root/'machine.json',
            phase_sdk=self.root/'sdk', phase_lut=self.root/'lut', mode='inspect')
        for name in ('checkpoint', 'machine_config', 'phase_lut'):
            getattr(self.args, name).touch()
        for name in ('processor', 'phase_sdk'):
            getattr(self.args, name).mkdir()
        self.args.protocol.write_text(json.dumps(dict(rows=self.rows)))
        self.geometry = dict(base_corners_screen_TL_TR_BR_BL=[[586,147],[1379,159],[1369,946],[573,932]],
            stage_calibration={s:dict(phase_candidate='hv_inverse', camera_orientation='flip_v') for s in runner.STAGES})
        self.args.geometry.write_text(json.dumps(self.geometry))

    def inspect(self):
        with patch.object(runner, 'digest', return_value=runner.CHECKPOINT):
            return runner.inspect_paths(self.args)

    def test_inspect_is_read_only(self):
        result = self.inspect()
        self.assertEqual(result['sample_count'], 2400)
        self.assertFalse(result['model_loaded'])
        self.assertFalse(result['devices_opened'])
        self.assertFalse(self.args.run_dir.exists())

    def test_wrong_checkpoint(self):
        self.args.checkpoint_sha256 = '0'*64
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA'): self.inspect()

    def test_wrong_checkpoint_bytes(self):
        with patch.object(runner, 'digest', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'checkpoint SHA'): runner.inspect_paths(self.args)

    def test_duplicate_identity(self):
        self.rows[-1]['sample_id'] = self.rows[0]['sample_id']
        with self.assertRaises(ValueError): runner.protocol_rows(dict(rows=self.rows), self.data)

    def test_unsafe_identity(self):
        self.rows[0]['sample_id'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'Unsafe'): runner.protocol_rows(dict(rows=self.rows), self.data)

    def test_image_path_escape(self):
        self.rows[0]['image_path'] = '../weight.pt'
        with self.assertRaisesRegex(ValueError, 'escapes'): runner.protocol_rows(dict(rows=self.rows), self.data)

    def test_missing_image(self):
        self.rows[0]['image_path'] = 'missing.png'
        with self.assertRaises(FileNotFoundError): runner.protocol_rows(dict(rows=self.rows), self.data)

    def test_extra_split(self):
        self.rows.append(dict(sample_id='f'*16, split='val', image_path='image.png'))
        with self.assertRaisesRegex(ValueError, 'Unexpected'): runner.protocol_rows(dict(rows=self.rows), self.data)

    def test_roi_and_orientation_guards(self):
        for edit in ('roi', 'orientation', 'stage'):
            g = json.loads(json.dumps(self.geometry))
            if edit == 'roi': g['base_corners_screen_TL_TR_BR_BL'][0][0] += 1
            elif edit == 'orientation': g['stage_calibration']['vision_router']['camera_orientation'] = 'identity'
            else: del g['stage_calibration']['vision_router']
            self.args.geometry.write_text(json.dumps(g))
            with self.subTest(edit=edit), self.assertRaises(ValueError): self.inspect()

    def test_completed_run_inspect_allowed_but_restart_refused(self):
        self.args.run_dir.mkdir()
        (self.args.run_dir/'report.json').write_text('{}')
        self.inspect()
        for mode in ('selftest', 'pilot', 'full'):
            self.args.mode = mode
            with self.subTest(mode=mode), self.assertRaises(FileExistsError): self.inspect()

    def test_main_default_inspect_does_not_import_model_or_open_sdk(self):
        argv = ['runner', '--checkpoint-sha256', runner.CHECKPOINT]
        for name in ('checkpoint','processor','protocol','data_root','geometry','run_dir','machine_config','phase_sdk','phase_lut'):
            argv.extend(['--'+name.replace('_','-'), str(getattr(self.args,name))])
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(('torch','transformers')) or name.endswith(('bench','model','replay')):
                raise AssertionError('Inspect attempted heavy/device import: '+name)
            return original(name, *args, **kwargs)
        with patch.object(sys, 'argv', argv), patch.object(runner,'digest',return_value=runner.CHECKPOINT), \
                patch('builtins.__import__',side_effect=guarded), patch('builtins.print') as output:
            runner.main()
        self.assertTrue(json.loads(output.call_args.args[0])['read_only'])
        self.assertFalse(self.args.run_dir.exists())

    def test_pipeline_explicit_callback_and_cpu_inputs(self):
        events=[]
        def picture(path, preprocessing):
            events.append(('picture',path,preprocessing)); return 'image'
        def inputs(processor, images, device):
            events.append(('inputs',images,device)); return 'batch'
        def replay(model, batch, ids, callback):
            self.assertEqual(batch,'batch')
            return callback('vision_router','active',ids)
        pipe=runner.Pipeline(self.data,inputs,picture,replay)
        pipe.STAGE_CALIBRATION={'vision_router':('hv_inverse','flip_v')}
        def capture(*args): events.append(('capture',args)); return 'result'
        pipe.capture_stage=capture
        with patch.object(runner,'torch',SimpleNamespace(device=lambda x:x),create=True):
            result=pipe.process_batch(SimpleNamespace(metadata={}),None,'bench','out',{'vision_router':'phase'},self.rows[:1])
        self.assertEqual(result,'result')
        self.assertEqual(events[1],('inputs',['image'],'cpu'))
        self.assertEqual(events[2][1],('bench','out','vision_router','phase','active',['0000000000000000'],'flip_v'))

    def test_exact_geometry_alias_is_bound(self):
        with patch.object(runner, 'phase_gray', return_value='gray') as phase:
            self.assertEqual(runner.selected_phase('radians','hv_inverse'),'gray')
        phase.assert_called_once_with('radians','hv',True)
        self.assertEqual(runner.math.pi, 3.141592653589793)

    def test_source_layer_order_warmup_and_lazy_sessions(self):
        tree=ast.parse(self.source())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loops=[n for n in ast.walk(main) if isinstance(n,ast.For)]
        self.assertTrue(any(ast.unparse(n.target)=='stage' and ast.unparse(n.iter)=='STAGES' for n in loops))
        self.assertTrue(any(ast.unparse(n.iter)=='range(2)' for n in loops))
        stagebench=next(n for n in ast.walk(main) if isinstance(n,ast.ClassDef) and n.name=='StageBench')
        enter=next(n for n in stagebench.body if isinstance(n,ast.FunctionDef) and n.name=='__enter__')
        self.assertEqual(ast.unparse(enter.body[0]),'return self')
        self.assertNotIn('SHSBench', ast.unparse(enter))

    def source(self):
        if hasattr(runner, '_test_source_text'): return runner._test_source_text
        return Path(runner.__file__).read_text(encoding='utf8')

    def test_lazy_session_opens_only_on_missing_capture_and_releases(self):
        tree=ast.parse(self.source())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        node=next(n for n in ast.walk(main) if isinstance(n,ast.ClassDef) and n.name=='StageBench')
        events=[]
        class FakeBench:
            def __init__(self,*args,**kwargs): events.append(('construct',args,kwargs))
            def __enter__(self): events.append('enter'); return self
            def __exit__(self,*exc): events.append(('exit',exc))
            def capture(self,*args,**kwargs): events.append(('capture',args,kwargs)); return 'ccd'
        namespace=dict(SHSBench=FakeBench,out='output',phase_paths={'stage':'phase'},args=self.args,
                       current=['vision_router'],json=json,print=lambda *a,**k:None)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<lazy-session>','exec'),namespace)
        with namespace['StageBench']() as bench:
            self.assertEqual(events,[])
            self.assertEqual(bench.capture('first'),'ccd')
            bench.capture('second')
            self.assertEqual(sum(isinstance(e,tuple) and e[0]=='construct' for e in events),1)
            bench.release()
            bench.release()
            self.assertEqual(sum(isinstance(e,tuple) and e[0]=='exit' for e in events),1)
            bench.capture('third')
        self.assertEqual(sum(isinstance(e,tuple) and e[0]=='construct' for e in events),2)
        self.assertEqual(sum(isinstance(e,tuple) and e[0]=='exit' for e in events),2)
        self.assertEqual(events[0][1][:3],('output',400,240))

    def test_partial_pair_refused_before_capture_or_model(self):
        tree=ast.parse(self.source())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
        folder=self.root/'ccd'/'vision_router'
        folder.mkdir(parents=True)
        (folder/(self.rows[0]['sample_id']+'.png')).touch()
        namespace=dict(STAGES=runner.STAGES,current=['vision_router'],out=self.root)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<partial-pair>','exec'),namespace)
        with self.assertRaisesRegex(RuntimeError,'Partial CCD pair'):
            namespace['capture'](None,None,'vision_router',None,None,[self.rows[0]['sample_id']],'flip_v')


if __name__ == '__main__': unittest.main()
