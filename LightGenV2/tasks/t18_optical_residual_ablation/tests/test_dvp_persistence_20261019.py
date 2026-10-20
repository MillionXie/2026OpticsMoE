"""No SDK/Torch: byte identity, buffer ownership and bounded failure semantics."""
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dvp_persistence',HERE/'lab_caltech20_dvp_persistence_20261019.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(HERE))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.addCleanup(gc.collect)
        self.raw=np.random.default_rng(17).integers(0,80,(512,640),np.uint8);self.crop=self.raw[:478,:478].copy()

    def test_identical_serial_PNG_bytes_and_SHA(self):
        Image.fromarray(self.raw).save(self.root/'serial_raw.png',compress_level=1);Image.fromarray(self.crop).save(self.root/'serial.png',compress_level=1)
        with m.FrameWriter() as writer:
            writer.submit(self.root/'async',self.raw,self.crop);row=writer.finish()
        self.assertEqual((self.root/'serial_raw.png').read_bytes(),(self.root/'async_raw.png').read_bytes())
        self.assertEqual((self.root/'serial.png').read_bytes(),(self.root/'async.png').read_bytes())
        self.assertEqual(row['raw_SHA256'],hashlib.sha256((self.root/'serial_raw.png').read_bytes()).hexdigest())

    def test_SDK_buffer_reuse_cannot_mutate_snapshot(self):
        expected_raw=self.raw.copy();expected_crop=self.crop.copy()
        with m.FrameWriter() as writer:
            writer.submit(self.root/'owned',self.raw,self.crop);self.raw.fill(255);self.crop.fill(255);writer.finish()
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'owned_raw.png')),expected_raw))
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'owned.png')),expected_crop))

    def test_queue_cannot_grow(self):
        with m.FrameWriter() as writer:
            writer.submit(self.root/'first',self.raw,self.crop)
            with self.assertRaises(AssertionError):writer.submit(self.root/'second',self.raw,self.crop)
            writer.finish();writer.submit(self.root/'second',self.raw,self.crop);writer.finish()

    def test_signal_exception_still_saves_bad_frame(self):
        with self.assertRaisesRegex(RuntimeError,'signal'):
            with m.FrameWriter() as writer:
                writer.submit(self.root/'bad',self.raw,self.crop);raise RuntimeError('signal')
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'bad_raw.png')),self.raw))

    def test_IO_failure_is_not_suppressed(self):
        with self.assertRaises(FileNotFoundError):
            with m.FrameWriter() as writer:writer.submit(self.root/'missing/frame',self.raw,self.crop);writer.finish()

    def test_existing_capture_is_never_overwritten(self):
        path=self.root/'old_raw.png';path.write_bytes(b'protected')
        with self.assertRaises(AssertionError):
            with m.FrameWriter() as writer:writer.submit(self.root/'old',self.raw,self.crop);writer.finish()
        self.assertEqual(path.read_bytes(),b'protected')

    def capture_fixture(self,name,overlap,bad_index=None,full=False,record_only=False,bad_value=2):
        root=self.root/name;root.mkdir();session=root/'L4';session.mkdir()
        lut=root/'lut.json';lut.write_text(json.dumps(dict(drive=list(range(256)))))
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        save=lambda p,v:Path(p).write_text(json.dumps(v))
        scope=dict(np=np,Path=Path,Image=Image,os=__import__('os'),time=time,json=json,shutil=__import__('shutil'),traceback=__import__('traceback'),sha=digest,save=save,read=lambda p:json.loads(Path(p).read_text()),__file__=str(HERE/'lab_caltech20_dvp_layerwise_20261019.py'),HARDWARE='hardware',LOCK=root/'lock',git_head=lambda:'test_source',model_for=lambda *args:(None,None),native=lambda x,k:x,phase_native=lambda x:x,corners=lambda p:[[0,0],[477,0],[477,477],[0,477]],orient=lambda x,k:x)
        calls=[]
        class Bench:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def stage(self,*args):pass
            def capture(self,path):
                index=len(calls);calls.append(index)
                raw=np.asarray(Image.open(path)).copy()
                if index==bad_index:
                    if bad_value=='transport':raise RuntimeError('injected SDK frame transport failure')
                    raw.fill(bad_value)
                return raw,dict(capture_total_ms=1,frame_id=index,exposure_us=5000.5,gain=1)
        scope['Bench']=Bench
        def padded(data,role,begin):
            end=min(begin+8,len(data[role][2]));return np.arange(begin,end),end-begin,end
        def forward(model,x,ids,stage,cache,physical):
            amp=np.stack([np.full((478,478),50+(int(i) if i!='__padding__' else 0)%10,np.uint8).astype(np.float32)/255 for i in ids])
            return physical(stage,amp,np.zeros((478,478),np.uint8),ids)
        scope.update(padded=padded,layer_forward=forward)
        tree=ast.parse((HERE/'lab_caltech20_dvp_layerwise_20261019.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='acquire');node.decorator_list=[]
        helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('signal_flags','session_for')];exec(compile(ast.Module(body=helpers,type_ignores=[]),'flags_fixture','exec'),scope)
        scope['validate_capture_policy']=lambda *args:dict(policy='validated_by_separate_policy_tests')
        scope['session_hardware_digest']=lambda sequence:'hardware'
        if full:
            class SmallCounts(ast.NodeTransformer):
                def visit_Constant(self,node):
                    if type(node.value) is int and node.value in (1600,411,2011):return ast.copy_location(ast.Constant(value={1600:16,411:9,2011:25}[node.value]),node)
                    return node
            node=SmallCounts().visit(node)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'capture_fixture','exec'),scope)
        data=dict(train=(np.zeros(24),np.zeros(24),np.array([str(i) for i in range(24)])))
        if full:data=dict(train=(np.zeros(16),np.zeros(16),np.array([str(i) for i in range(16)])),test=(np.zeros(9),np.zeros(9),np.array([str(i) for i in range(16,25)])))
        a=SimpleNamespace(sequence=root,depth=4,async_png=overlap)
        if record_only:a.capture_policy=lut
        error=None
        try:scope['acquire'](a,'router',root/'unused.pt','pt',data,dict(sha256='data'),dict(response_lut=str(lut),coordinates='unused',orientation='identity'),pilot=not full)
        except BaseException as e:error=e.with_traceback(None)
        self.assertFalse((root/'lock').exists())
        return session/('capture/router' if full else 'router_pilot24'),calls,error

    def test_actual_capture_control_flow_serial_vs_overlap(self):
        serial,scalls,se=self.capture_fixture('serial',False);overlap,acalls,ae=self.capture_fixture('overlap',True)
        self.assertIsNone(se);self.assertIsNone(ae);self.assertEqual(scalls,acalls)
        sr=json.loads((serial/'report.json').read_text());ar=json.loads((overlap/'report.json').read_text())
        self.assertEqual(sr['arrays'],ar['arrays']);self.assertEqual(ar['physical_ccd'],24)
        for i in range(24):
            s=json.loads((serial/'train'/f'{i:06d}.json').read_text());a=json.loads((overlap/'train'/f'{i:06d}.json').read_text())
            for k in ('sample_id','index','role','PT_SHA256','phase_SHA256','canonical_SHA256','input_SHA256','raw_SHA256','p99','mean','saturation_fraction'):self.assertEqual(s[k],a[k])
            self.assertEqual((serial/'train'/f'{i:06d}_raw.png').read_bytes(),(overlap/'train'/f'{i:06d}_raw.png').read_bytes())

    def test_actual_capture_signal_guard_drains_receipt(self):
        out,calls,error=self.capture_fixture('bad_capture',True,6)
        self.assertIsInstance(error,RuntimeError);self.assertEqual(len(calls),7)
        row=json.loads((out/'train/000006.json').read_text());self.assertEqual(row['p99'],2)
        self.assertTrue((out/'train/000006_raw.png').exists());self.assertEqual(json.loads((out/'failure.json').read_text())['frames'],7)

    def test_record_only_dark_frame_preserved_and_capture_continues(self):
        out,calls,error=self.capture_fixture('record_dark',True,6,record_only=True)
        self.assertIsNone(error);self.assertEqual(len(calls),24)
        row=json.loads((out/'train/000006.json').read_text());self.assertEqual(row['p99'],2)
        self.assertTrue(row['signal_threshold_flags']['raw_p99_below15']);self.assertEqual(row['signal_threshold_action'],'record_only')
        self.assertTrue((out/'train/000006_raw.png').exists());self.assertTrue((out/'report.json').exists())

    def test_record_only_saturated_frame_preserved_and_capture_continues(self):
        out,calls,error=self.capture_fixture('record_clip',True,6,record_only=True,bad_value=255)
        self.assertIsNone(error);self.assertEqual(len(calls),24)
        row=json.loads((out/'train/000006.json').read_text());self.assertEqual(row['saturation_fraction'],1.)
        self.assertTrue(row['signal_threshold_flags']['saturation_above15percent']);self.assertTrue((out/'train/000006_raw.png').exists())

    def test_record_only_cannot_ignore_transport_failure(self):
        out,calls,error=self.capture_fixture('record_transport',True,6,record_only=True,bad_value='transport')
        self.assertIsInstance(error,RuntimeError);self.assertIn('SDK frame',str(error));self.assertFalse((out/'report.json').exists())

    def test_actual_TRAIN_TEST_role_boundary_and_padding(self):
        serial,_,se=self.capture_fixture('full_serial',False,full=True);overlap,calls,ae=self.capture_fixture('full_overlap',True,full=True)
        self.assertIsNone(se);self.assertIsNone(ae);self.assertEqual(len(calls),25)
        sr=json.loads((serial/'report.json').read_text());ar=json.loads((overlap/'report.json').read_text());self.assertEqual(sr['arrays'],ar['arrays'])
        self.assertEqual(ar['TRAIN'],16);self.assertEqual(ar['TEST'],9)
        for role,n in (('train',16),('test',9)):
            for i in range(n):
                r=json.loads((overlap/role/f'{i:06d}.json').read_text());self.assertEqual(r['role'],role);self.assertEqual(r['index'],i)
                self.assertEqual((serial/role/f'{i:06d}_raw.png').read_bytes(),(overlap/role/f'{i:06d}_raw.png').read_bytes())

    def test_previous_IO_failure_preserves_already_captured_next_frame(self):
        import lab_caltech20_dvp_persistence_20261019 as actual
        original=actual.write_images
        def failed_first(base,raw,crop):
            if Path(base).name=='000000':raise OSError('injected previous IO error')
            return original(base,raw,crop)
        with patch.object(actual,'write_images',failed_first):out,calls,error=self.capture_fixture('io_failure',True)
        self.assertIsInstance(error,OSError);self.assertEqual(len(calls),2)
        r=json.loads((out/'following_frame_rescue.json').read_text());self.assertEqual(r['index'],1)
        self.assertTrue((out/'train/000001_raw.png').exists());self.assertTrue((out/'persistence_failure.json').exists())


if __name__=='__main__':unittest.main()
