"""Recovery identity/known-pattern tests without importing Torch or opening SDKs."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


HERE=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value))


def functions(path,names,scope):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.scope=dict(np=np,Path=Path,Image=Image,sha=sha,read=read,save=save,HARDWARE='hardware')
        functions(HERE/'lab_caltech20_dvp_20261019.py',('variants','orient','correlation'),self.scope)
        functions(HERE/'lab_caltech20_dvp_layerwise_20261019.py',('session_hardware_digest','admitted_pilot','revalidate_replug','validate_stage_hardware','validate_fit_ancestry'),self.scope)
        functions(HERE/'lab_caltech20_dvp_stage_exposure_20261019.py',('exposure_passes','contrast_statistics','contrast_passes'),self.scope)

    def pilot(self):
        folder=self.root/'router_pilot24';(folder/'train').mkdir(parents=True)
        Image.fromarray(np.zeros((478,478),np.uint8)).save(folder/'phase.bmp')
        array=np.zeros((24,478,478),np.uint8);np.save(folder/'train/ccd.npy',array)
        for i in range(24):
            base=folder/'train'/f'{i:06d}'
            for suffix in ('_input.png','_raw.png','.png'):Image.fromarray(array[i]).save(base.with_name(base.name+suffix))
            meta=dict(exposure_us=5000.5,gain=1,shape=[3648,5480],format=0,bits=0,flip_h=False,flip_v=False,rotate=False,opposite=False)
            save(base.with_suffix('.json'),dict(sample_id=str(i),index=i,role='train',stage='router',PT_SHA256='pt',hardware_SHA256='hardware',data_SHA256='data',phase_SHA256=sha(folder/'phase.bmp'),canonical_SHA256=sha(base.with_name(base.name+'_input.png')),raw_SHA256=sha(base.with_name(base.name+'_raw.png')),p99=30,saturation_fraction=0,frame_metadata=meta))
        save(folder/'report.json',dict(state='complete',pilot=True,physical_ccd=24,sdk_released=True,PT_SHA256='pt',hardware_SHA256='hardware',data_SHA256='data',arrays=dict(train=dict(SHA256=sha(folder/'train/ccd.npy')))))
        return dict(train=(None,None,np.array([str(i) for i in range(24)]))),dict(sha256='data')

    def test_exact_pilot_and_changed_checkpoint(self):
        data,dm=self.pilot();_,array=self.scope['admitted_pilot'](self.root,data,dm,'pt');self.assertEqual(array.shape,(24,478,478))
        with self.assertRaises(AssertionError):self.scope['admitted_pilot'](self.root,data,dm,'other_pt')

    def test_changed_raw_rejected(self):
        data,dm=self.pilot();Image.fromarray(np.ones((478,478),np.uint8)).save(self.root/'router_pilot24/train/000003_raw.png')
        with self.assertRaises(AssertionError):self.scope['admitted_pilot'](self.root,data,dm,'pt')

    def validation(self,brightness=1,wrong_direction=False):
        response=self.root/'response.json';save(response,dict(fit_means=[0,120]));hardware=dict(response_lut=str(response),orientation='flip_v')
        self.scope.update(native=lambda x,kind:x,phase_native=lambda x:x)
        class Bench:
            settings=dict(exposure_us=5000.5,gain=1.)
            def stage(self,path):pass
            def capture(self,path):
                pixels=np.asarray(Image.open(path));signal=np.rint(pixels.astype(float)**2/255**2*120*brightness).clip(0,255).astype(np.uint8)
                return signal if wrong_direction else np.flipud(signal),dict(frame_id=1)
        self.scope['revalidate_replug'](Bench(),self.root,hardware,np.eye(3))

    def test_known_patterns_pass(self):self.validation();self.assertTrue(read(self.root/'replug_validation/report.json')['passed'])
    def test_brightness_drift_rejected(self):
        with self.assertRaises(AssertionError):self.validation(brightness=.7)
    def test_wrong_direction_rejected(self):
        with self.assertRaises(AssertionError):self.validation(wrong_direction=True)

    def exposure(self,p99=30,saturation=0,response_error=0,dark_p99=3):
        cfg=dict(TRAIN_probe_samples=40,probe_min_p99=25,probe_max_saturation=.01,max_dark_p99=8,linearity_absolute_gray=2,linearity_relative_tolerance=.15)
        models=[dict(p99=p99,saturation_fraction=saturation) for _ in range(40)]
        controls=[dict(tag='dark',mean=2,p99=dark_p99,saturation_fraction=0),dict(tag='gray64',mean=10+response_error,saturation_fraction=0),dict(tag='gray96',mean=26+response_error,saturation_fraction=0)]
        return self.scope['exposure_passes'](models,controls,dict(dark=2,gray64=4,gray96=8),4,cfg)

    def test_exposure_admission_pass(self):self.assertTrue(self.exposure())
    def test_exposure_low_signal_rejected(self):self.assertFalse(self.exposure(p99=14))
    def test_exposure_clip_rejected(self):self.assertFalse(self.exposure(saturation=.02))
    def test_exposure_nonlinear_response_rejected(self):self.assertFalse(self.exposure(response_error=20))
    def test_exposure_noise_floor_rejected(self):self.assertFalse(self.exposure(dark_p99=25))

    def contrast(self,contrast=30,snr=10,background_p99=20,clip=0,error=0):
        cfg=dict(TRAIN_probe_samples=40,probe_min_p99=25,probe_max_saturation=.01,background_max_p99=64,probe_min_contrast_p99=25,probe_min_SNR=5,linearity_absolute_gray=2,linearity_relative_tolerance=.15)
        rows=[dict(p99=50,saturation_fraction=clip,contrast_p99=contrast,contrast_SNR=snr) for _ in range(40)]
        controls=[dict(tag='dark',mean=10),dict(tag='gray64',mean=18+error,saturation_fraction=0),dict(tag='gray96',mean=34+error,saturation_fraction=0)]
        return self.scope['contrast_passes'](rows,controls,dict(dark=2,gray64=4,gray96=8),4,dict(p99=background_p99,saturation_fraction=0),cfg)

    def test_contrast_has_independent_admission(self):self.assertTrue(self.contrast())
    def test_contrast_low_signal_rejected(self):self.assertFalse(self.contrast(contrast=24))
    def test_contrast_noise_rejected(self):self.assertFalse(self.contrast(snr=4))
    def test_contrast_headroom_rejected(self):self.assertFalse(self.contrast(background_p99=65))
    def test_contrast_clipping_rejected(self):self.assertFalse(self.contrast(clip=.02))
    def test_contrast_linearity_rejected(self):self.assertFalse(self.contrast(error=15))
    def test_contrast_does_not_modify_raw_array(self):
        crop=np.full((478,478),50,np.uint8);background=np.full((478,478),20,np.float32)
        result=self.scope['contrast_statistics'](crop,background,2)
        self.assertEqual(result,dict(contrast_p99=30,contrast_SNR=15));self.assertTrue((crop==50).all());self.assertTrue((background==20).all())

    def test_remaining_profiles_preserve_admitted_criteria(self):
        original=read(HERE/'lab_caltech20_DVP_stage_contrast_profile_20261019.json')
        remaining=read(HERE/'lab_caltech20_DVP_remaining_contrast_profile_20261019.json')
        for k in original:
            if k not in ('scope','exposure_probe_us'):self.assertEqual(original[k],remaining[k])
        self.assertEqual(remaining['exposure_probe_us'],[5000,10000,20000,40000,80000])
        fit=read(HERE/'lab_caltech20_DVP_remaining_adapt_profile_20261019.json')
        self.assertEqual(sha(HERE/fit['base_profile']),fit['base_profile_SHA256'])
        self.assertEqual(fit['overrides']['teacher_simulated_newest_layers'],1)
        recovery=read(HERE/'lab_caltech20_DVP_stage0_recovery_profile_20261019.json')
        for key in ('phase_lr','electronic_lr','alpha_lr','head_lr','label_smoothing','confusion_emphasis','teacher_weight','middle_fit'):
            self.assertEqual(fit['overrides'][key],recovery['overrides'][key])
        self.assertEqual(fit['overrides']['middle_fit'],dict(epochs=24,seconds=1000))
        self.assertEqual(fit['overrides']['final_fit'],dict(epochs=120,seconds=1200))

    def test_stage2_mid_exposure_preserves_every_threshold(self):
        original=read(HERE/'lab_caltech20_DVP_remaining_contrast_profile_20261019.json')
        bounded=read(HERE/'lab_caltech20_DVP_L4_stage2_mid60_profile_20261019.json')
        self.assertEqual(bounded['exposure_probe_us'],[60000])
        for key in original:
            if key not in ('scope','exposure_probe_us'):self.assertEqual(original[key],bounded[key])

    def hardware(self):
        base=self.root/'calibration/admission/admitted_hardware.json';base.parent.mkdir(parents=True)
        keys=('camera_dll','camera_dll_SHA256','coordinates','coordinates_SHA256','sensor_shape','machine_config','amplitude_sdk','phase_sdk','phase_lut','phase_LUT_SHA256','gain','wait_ms','discard_frames','phase_transform','orientation','response_lut','response_lut_SHA256')
        original={k:k for k in keys};original.update(gain=1,exposure_us=5000.5);save(base,original);self.scope['HARDWARE']=sha(base)
        report=self.root/'report.json';save(report,dict(passed=True,SDK_released=True,TEST_used=False,TRAIN=40,selected=dict(actual=dict(exposure_us=40000),probe_directory='probes'),PT_SHA256='pt',phase_SHA256='phase',data_SHA256='data'))
        h=dict(original,base_admission_SHA256=sha(base),exposure_us=40000,stage_admission_report=str(report),stage_admission_report_SHA256=sha(report),PT_SHA256='pt',phase_SHA256='phase',data_SHA256='data',probe_directory='probes',probe_indices=list(range(40)))
        path=self.root/'hardware.json';save(path,h);return path,h

    def test_new_exposure_hardware_identity(self):
        path,_=self.hardware();self.assertEqual(self.scope['validate_stage_hardware'](path,self.root)['exposure_us'],40000)
    def test_exposure_cannot_change_coordinates(self):
        path,h=self.hardware();h['coordinates']='other';save(path,h)
        with self.assertRaises(AssertionError):self.scope['validate_stage_hardware'](path,self.root)
    def test_exposure_cannot_substitute_probe_identity(self):
        path,h=self.hardware();h['probe_directory']='other';save(path,h)
        with self.assertRaises(AssertionError):self.scope['validate_stage_hardware'](path,self.root)

    def ancestry(self):
        identities={'router/train':'array_r','0/train':'array_e'}
        def node(name,payload,parent):
            folder=self.root/'fits'/name;folder.mkdir(parents=True);(folder/'best.pt').write_bytes(payload);digest=sha(folder/'best.pt')
            save(folder/'report.json',dict(state='complete',frozen_parameters_unchanged=True,results=dict(best=dict(PT_SHA256=digest,strict_CPU=True))))
            save(folder/'protocol.json',dict(prefix=2,prefix_SHA256=identities,data_SHA256='data',parent_PT_SHA256=parent));return digest
        first=node('0_exposure',b'first','capture_parent');last=node('0_recovery',b'second',first)
        return identities,first,last

    def test_multi_refinement_ancestry_has_no_canonical_alias_dependency(self):
        identities,first,last=self.ancestry();chain=self.scope['validate_fit_ancestry'](self.root,'0_recovery',last,2,'capture_parent',identities,'data')
        self.assertEqual([r['fit'] for r in chain],['0_recovery','0_exposure'])
    def test_ancestry_rejects_changed_prefix_arrays(self):
        identities,_,last=self.ancestry()
        with self.assertRaises(AssertionError):self.scope['validate_fit_ancestry'](self.root,'0_recovery',last,2,'capture_parent',dict(identities,extra='bad'),'data')
    def test_ancestry_rejects_changed_parent_PT(self):
        identities,_,last=self.ancestry();(self.root/'fits/0_exposure/best.pt').write_bytes(b'changed')
        with self.assertRaises(AssertionError):self.scope['validate_fit_ancestry'](self.root,'0_recovery',last,2,'capture_parent',identities,'data')
    def test_ancestry_rejects_cycle(self):
        identities,_,last=self.ancestry();p=self.root/'fits/0_exposure/protocol.json';r=read(p);r['parent_PT_SHA256']=last;save(p,r)
        with self.assertRaises(AssertionError):self.scope['validate_fit_ancestry'](self.root,'0_recovery',last,2,'capture_parent',identities,'data')


if __name__=='__main__':unittest.main()
