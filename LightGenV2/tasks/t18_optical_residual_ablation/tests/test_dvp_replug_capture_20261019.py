"""No SDK/Torch: one-connection recheck, strict response guards and zero-frame archive."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parents[1]


class ReplugCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.save=lambda p,v:Path(p).write_text(json.dumps(v))
        self.read=lambda p:json.loads(Path(p).read_text())
        self.tree=ast.parse((HERE/'lab_caltech20_dvp_layerwise_20261019.py').read_text())

    def test_shared_policy_is_fixed_and_explicit_not_a_signal_claim(self):
        cfg=self.read(HERE/'lab_caltech20_DVP_shared_capture_profile_20261019.json')
        self.assertEqual(cfg['exposure_us'],60006.);self.assertEqual(cfg['gain'],1.)
        self.assertFalse(cfg['calibrate_each_stage']);self.assertEqual(cfg['brightness_threshold_action'],'record_only')
        self.assertEqual(cfg['saturation_threshold_action'],'record_only');self.assertFalse(cfg['network_background_subtraction'])

    def test_signal_flags_record_even_fully_dark_or_saturated(self):
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='signal_flags');scope={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'signal_fixture','exec'),scope)
        flags=scope['signal_flags'](dict(p99=0,saturation_fraction=1.,contrast_p99=0),dict(capture_min_contrast_p99=15))
        self.assertEqual(flags,dict(raw_p99_below15=True,saturation_above15percent=True,net_p99_below_limit=True))

    def test_optical_threshold_raises_require_strict_policy(self):
        acquire=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='acquire')
        guards=[n for n in ast.walk(acquire) if isinstance(n,ast.If) and any(isinstance(v,ast.Raise) and isinstance(v.exc,ast.Call) and v.exc.args and isinstance(v.exc.args[0],ast.Constant) and str(v.exc.args[0].value).startswith(('DVP signal guard:','DVP net-signal quality guard:')) for v in n.body)]
        self.assertEqual(len(guards),2)
        for guard in guards:self.assertIn('not record_only',ast.unparse(guard.test))

    def test_shared_policy_validates_existing_calibration_without_camera(self):
        cfg=self.read(HERE/'lab_caltech20_DVP_shared_capture_profile_20261019.json');source=self.root/cfg['source_hardware'];source.parent.mkdir(parents=True);source.write_bytes(b'verified admission');cfg['source_hardware_SHA256']=self.sha(source)
        profile=self.root/'profile.json';self.save(profile,cfg)
        scope=dict(read=self.read,Path=Path,sha=self.sha,HARDWARE=cfg['base_admission_SHA256'],session_hardware_digest=lambda sequence:cfg['base_admission_SHA256'],validate_stage_hardware=lambda *args:dict(quality_mode='same_phase_contrast',exposure_us=60006.,gain=1.))
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_capture_policy');exec(compile(ast.Module(body=[node],type_ignores=[]),'policy_fixture','exec'),scope)
        hardware=dict(exposure_us=60006.,gain=1.);self.assertEqual(scope['validate_capture_policy'](profile,hardware,self.root),cfg)
        for key,value in [('calibrate_each_stage',True),('exposure_us',5000.5),('brightness_threshold_action','stop')]:
            altered=dict(cfg);altered[key]=value;self.save(profile,altered)
            with self.assertRaises(AssertionError):scope['validate_capture_policy'](profile,hardware,self.root)

    def test_fixed_hardware_preparation_has_no_camera_or_calibration_calls(self):
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_fixed_hardware');calls={ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)}
        self.assertFalse(calls&{'Bench','Camera','calibration','revalidate_replug'})
        text=ast.unparse(node);self.assertIn('phase_SHA256=sha(phase)',text);self.assertIn('probe_indices=[]',text)

    def geometry_identity(self):
        coordinate=self.root/'coordinates.txt';coordinate.write_bytes(b'new geometry')
        previous=dict(coordinates='old.txt',coordinates_SHA256='old',exposure_us=5000.5,gain=1.,orientation='flip_v',response_lut='frozen LUT')
        old=self.root/'original_admission.json';self.save(old,previous);base=self.root/'calibration/admission/admitted_hardware.json';base.parent.mkdir(parents=True)
        current=dict(previous,coordinates=str(coordinate),coordinates_SHA256=self.sha(coordinate),exposure_us=60006.);self.save(base,current)
        identity=self.root/'hardware_identity.json';record=dict(schema='fresh_DVP_geometry_v1',source_admission=str(old),source_admission_SHA256=self.sha(old),hardware_SHA256=self.sha(base),coordinates_SHA256=self.sha(coordinate),reuse_previous_CCD=False,fresh_layerwise_capture=True);self.save(identity,record)
        scope=dict(Path=Path,read=self.read,sha=self.sha,HARDWARE=self.sha(old));node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='session_hardware_digest');exec(compile(ast.Module(body=[node],type_ignores=[]),'identity_fixture','exec'),scope)
        return scope['session_hardware_digest'],base,coordinate,identity,record

    def test_fresh_coordinate_identity_explicit_and_immutable(self):
        fn,base,coordinate,identity,record=self.geometry_identity();self.assertEqual(fn(self.root),self.sha(base));coordinate.write_bytes(b'changed later')
        with self.assertRaises(AssertionError):fn(self.root)

    def test_new_geometry_cannot_adopt_old_CCD(self):
        fn,base,coordinate,identity,record=self.geometry_identity();record['reuse_previous_CCD']=True;self.save(identity,record)
        with self.assertRaises(AssertionError):fn(self.root)

    def test_geometry_identity_cannot_change_orientation_or_LUT(self):
        fn,base,coordinate,identity,record=self.geometry_identity();current=self.read(base);current['orientation']='identity';self.save(base,current);record['hardware_SHA256']=self.sha(base);self.save(identity,record)
        with self.assertRaises(AssertionError):fn(self.root)

    def test_geometry_identity_cannot_hide_exposure_search(self):
        fn,base,coordinate,identity,record=self.geometry_identity();current=self.read(base);current['exposure_us']=80000.;self.save(base,current);record['hardware_SHA256']=self.sha(base);self.save(identity,record)
        with self.assertRaises(AssertionError):fn(self.root)

    def geometry_check_fixture(self,actual_orientation='flip_v'):
        lut=self.root/'lut.json';self.save(lut,dict(drive=list(range(256))));calls=[]
        class FakeBench:
            settings=dict(exposure_us=60006.,gain=1.)
            def stage(self,path):calls.append(('stage',str(path)))
            def capture(self,path):
                calls.append(('capture',str(path)));gray=np.asarray(Image.open(path));raw=np.flipud(gray).copy() if actual_orientation=='flip_v' else gray.copy()
                return raw,dict(frame_id=len(calls),exposure_us=60006.,gain=1.)
        scope=dict(np=np,Path=Path,Image=Image,sha=self.sha,read=self.read,save=self.save,native=lambda x,k:x,phase_native=lambda x:x)
        nodes=[n for n in ast.parse((HERE/'lab_caltech20_dvp_20261019.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('variants','orient','correlation')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'direction_fixture','exec'),scope)
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='shared_geometry_check');exec(compile(ast.Module(body=[node],type_ignores=[]),'geometry_fixture','exec'),scope)
        return scope['shared_geometry_check'],FakeBench(),dict(response_lut=str(lut),orientation='flip_v',coordinates_SHA256='new_coordinates'),calls

    def test_two_geometry_patterns_same_connection_no_exposure_search(self):
        fn,bench,hardware,calls=self.geometry_check_fixture();r=fn(bench,self.root/'geometry',hardware,np.eye(3,dtype=np.float32));self.assertTrue(r['passed']);self.assertFalse(r['camera_reopened']);self.assertFalse(r['exposure_search']);self.assertEqual(len([c for c in calls if c[0]=='capture']),2)

    def test_wrong_geometry_direction_fails_with_raw_evidence(self):
        fn,bench,hardware,calls=self.geometry_check_fixture(actual_orientation='identity')
        with self.assertRaisesRegex(AssertionError,'direction/timing'):fn(bench,self.root/'geometry',hardware,np.eye(3,dtype=np.float32))
        self.assertFalse(self.read(self.root/'geometry/report.json')['passed']);self.assertEqual(len(list((self.root/'geometry').glob('*_raw.png'))),2)

    def fixture(self,level=55,noise=0,failed_known=False,wrong_phase=False):
        phase=self.root/'model_phase.bmp';phase.write_bytes(b'exact frozen phase')
        lut=self.root/'lut.json';self.save(lut,dict(drive=list(range(256))))
        prior=self.root/'background.npy';np.save(prior,np.full((478,478),55,np.float32));before=self.sha(prior)
        hardware=dict(exposure_us=60006.,gain=1.,stage_admission_report=str(self.root/'admission.json'),phase_SHA256='wrong' if wrong_phase else self.sha(phase),response_lut=str(lut),quality_background=dict(path=str(prior),noise_sigma=.5822),orientation='identity')
        original=dict(exposure_us=5000.5,discard_frames=6);calls=[]
        class FakeBench:
            def __init__(self):self.settings=dict(exposure_us=60006.,gain=1.);self.camera=self;self.frame=0;self.captures=0
            def read(self,check_settings=True):self.frame+=1;calls.append(('discard',self.settings['exposure_us']))
            def stage(self,path):calls.append(('stage',str(path)))
            def capture(self,path):
                self.frame+=1;self.captures+=1;offset=noise if self.captures%2 else -noise;calls.append(('capture',id(self),self.settings['exposure_us']))
                return np.full((478,478),level+offset,np.uint8),dict(frame_id=self.frame,exposure_us=self.settings['exposure_us'],gain=1.)
        bench=FakeBench()
        class Camera:
            def settings(self,exposure,gain):bench.settings=dict(exposure_us=exposure,gain=gain);calls.append(('settings',exposure));return bench.settings.copy()
        bench.camera=Camera()
        def known(current,folder,cfg,warp):
            self.assertIs(current,bench);calls.append(('known',id(current),current.settings['exposure_us']))
            dest=folder/'replug_validation';dest.mkdir();self.save(dest/'report.json',dict(passed=not failed_known))
            if failed_known:raise RuntimeError('known pattern failed')
        scope=dict(np=np,Path=Path,Image=Image,sha=self.sha,read=self.read,save=self.save,native=lambda x,k:x,orient=lambda x,k:x,revalidate_replug=known)
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='same_connection_replug_check')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'replug_fixture','exec'),scope)
        return scope['same_connection_replug_check'],bench,hardware,original,calls,prior,before

    def test_same_object_original_check_then_selected_exposure(self):
        fn,bench,hw,original,calls,prior,before=self.fixture();result=fn(bench,self.root,hw,original,np.eye(3,dtype=np.float32))
        self.assertTrue(result['passed']);self.assertFalse(result['camera_reopened']);self.assertEqual(bench.captures,2)
        self.assertEqual([c[1] for c in calls if c[0]=='settings'],[5000.5,60006.])
        self.assertEqual(next(c for c in calls if c[0]=='known')[2],5000.5)
        self.assertEqual({c[1] for c in calls if c[0]=='capture'},{id(bench)})
        self.assertEqual(self.sha(prior),before);self.assertTrue(result['prior_background_unchanged'])
        self.assertFalse(result['network_background_subtraction']);self.assertEqual(len(list(self.root.rglob('*_raw.png'))),2)

    def test_background_above_original_limit_rejected_with_evidence(self):
        fn,bench,hw,original,*_=self.fixture(level=65)
        with self.assertRaisesRegex(AssertionError,'changed response'):fn(bench,self.root,hw,original,np.eye(3,dtype=np.float32))
        r=self.read(self.root/'same_connection_replug_check/report.json');self.assertFalse(r['passed']);self.assertEqual(bench.captures,2)
        self.assertEqual(len(list(self.root.rglob('*_raw.png'))),2)

    def test_response_drift_rejected_even_below_background_limit(self):
        fn,bench,hw,original,*_=self.fixture(level=59)
        with self.assertRaises(AssertionError):fn(bench,self.root,hw,original,np.eye(3,dtype=np.float32))
        self.assertEqual(self.read(self.root/'same_connection_replug_check/report.json')['background_RMSE'],4.)

    def test_new_noise_rejected_without_changing_background(self):
        fn,bench,hw,original,_,prior,before=self.fixture(noise=3)
        with self.assertRaises(AssertionError):fn(bench,self.root,hw,original,np.eye(3,dtype=np.float32))
        self.assertEqual(self.sha(prior),before)

    def test_wrong_phase_rejected_before_dark_capture(self):
        fn,bench,hw,original,*_=self.fixture(wrong_phase=True)
        with self.assertRaises(AssertionError):fn(bench,self.root,hw,original,np.eye(3,dtype=np.float32))
        self.assertEqual(bench.captures,0)

    def test_known_pattern_failure_cannot_enter_formal_capture(self):
        fn,bench,hw,original,*_=self.fixture(failed_known=True)
        with self.assertRaisesRegex(RuntimeError,'known pattern'):fn(bench,self.root,hw,original,np.eye(3,dtype=np.float32))
        self.assertEqual(bench.captures,0)

    def test_acquire_calls_recheck_inside_only_Bench_context_before_roles(self):
        fn=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='acquire')
        contexts=[n for n in ast.walk(fn) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='Bench' for i in n.items)]
        self.assertEqual(len(contexts),1);body=contexts[0].body
        ix=next(i for i,n in enumerate(body) if 'same_connection_replug_check(' in ast.unparse(n))
        role=next(i for i,n in enumerate(body) if isinstance(n,ast.For) and ast.unparse(n.target)=='role');self.assertLess(ix,role)
        helper=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='same_connection_replug_check')
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('Bench','Camera') for n in ast.walk(helper)))

    def archive_fixture(self,frames=0,digest='pt',occupied=False):
        session=self.root/'L4';out=session/'capture/2';out.mkdir(parents=True)
        self.save(out/'failure.json',dict(frames=frames,error="RuntimeError('DVP Start: -1001')"));self.save(out/'contract.json',dict(PT_SHA256='pt',hardware_SHA256='hardware'))
        if occupied:(out.parent/'2_failed_reopen_start_20261019').mkdir()
        acquire=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='acquire')
        guard=next(n for n in acquire.body if isinstance(n,ast.If) and 'archive_empty_start' in ast.unparse(n.test))
        scope=dict(LOCK=self.root/'lock',read=self.read,save=self.save)
        wrapper=ast.parse('def archive(a,out,session,stage,pilot,digest,hardware_digest):\n pass').body[0];wrapper.body=[guard]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'archive_fixture','exec'),scope)
        return lambda:scope['archive'](SimpleNamespace(depth=4,archive_empty_start=True,recheck_before_capture=True),out,session,'2',False,digest,'hardware'),out

    def test_only_exact_zero_frame_failure_archived_recoverably(self):
        fn,out=self.archive_fixture();fn();archive=out.with_name('2_failed_reopen_start_20261019')
        self.assertFalse(out.exists());self.assertTrue((archive/'failure.json').exists());self.assertFalse(self.read(archive/'archive_receipt.json')['deletion'])

    def test_nonempty_failure_cannot_be_archived(self):
        fn,out=self.archive_fixture(frames=1)
        with self.assertRaises(AssertionError):fn()
        self.assertTrue(out.exists())

    def test_wrong_parent_cannot_be_archived(self):
        fn,out=self.archive_fixture(digest='other')
        with self.assertRaises(AssertionError):fn()
        self.assertTrue(out.exists())

    def test_archive_target_never_overwritten(self):
        fn,out=self.archive_fixture(occupied=True)
        with self.assertRaises(AssertionError):fn()
        self.assertTrue(out.exists())


if __name__=='__main__':unittest.main()
