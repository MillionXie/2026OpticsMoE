"""Synthetic software regression only; these tests do not certify the optical bench."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
from phase_fingerprint import classify,fingerprint,pcc,validate_bank,digest
from guarded_workflow import Guard,require_geometry,load_bank,write
from guarded_batch import initialize,quarantine,assert_no_pending,accept
from phase_hdmi import sha

class FingerprintTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        self.a=rng.integers(20,160,(96,96)).astype(float);self.b=rng.integers(20,160,(96,96)).astype(float)
        self.s={'std':30.,'p99':160.,'saturation_fraction':0.,'mean':90.}
    def test_right_wrong_ambiguous(self):
        refs={'a':self.a,'b':self.b}
        self.assertTrue(classify(self.a,self.s,refs,'a')['passed'])
        self.assertFalse(classify(self.b,self.s,refs,'a')['passed'])
        self.assertFalse(classify(self.a,self.s,{'a':self.a,'b':self.a},'a')['passed'])
    def test_dim_saturated_and_brightness_drift(self):
        for s in ({**self.s,'std':0},{**self.s,'p99':0},{**self.s,'saturation_fraction':.1}):
            self.assertFalse(classify(self.a,s,{'a':self.a,'b':self.b},'a')['passed'])
        self.assertFalse(classify(self.a*2,self.s,{'a':self.a,'b':self.b},'a')['passed'])
        self.assertEqual(pcc(np.ones(10),np.ones(10)),-1)
    def test_bank_does_not_hide_first_failure(self):
        _,report=validate_bank({'a':[self.a]*3,'b':[self.b]*3},{'a':[self.s]*3,'b':[self.s]*3})
        self.assertTrue(report['passed'])
        _,report=validate_bank({'a':[self.b,self.a,self.a],'b':[self.b]*3},{'a':[self.s]*3,'b':[self.s]*3})
        self.assertFalse(report['passed'])
    def test_raw_fixed_scale_and_roi(self):
        a=np.arange(10000,dtype=np.uint16).reshape(100,100)%255
        v,s=fingerprint(a.astype(np.uint8),[0,0,100,100]);self.assertEqual(v.shape,(96,96))
        with self.assertRaises(ValueError):fingerprint(a,[0,0,100,100])
        with self.assertRaises(ValueError):fingerprint(a.astype(np.uint8),[-1,0,100,100])
    def test_geometry(self):
        c={'geometry_confirmed':True,'logical_corners_full_sensor_xy':dict(zip(['top_left','top_right','bottom_right','bottom_left'],[[100,100],[900,100],[900,900],[100,900]]))}
        self.assertEqual(require_geometry(c),[100,100,901,901])
        c['logical_corners_full_sensor_xy']['bottom_right']=[100,900]
        with self.assertRaises(ValueError):require_geometry(c)
    def test_guard_bounded_retry(self):
        owner=Mock();bank={'phase_paths':{'lens':'lens.bmp'},'bank_id':'bank'}
        g=Guard.__new__(Guard);g.owner=owner;g.bank=bank;g.link={'phase_max_attempts':2};g.out=Path('unused')
        ok={'passed':True,'receipt':{},'file':'x.png','scores':{}}
        g.check=Mock(side_effect=[ok,{'passed':False},ok,ok])
        r,_=g.ensure('stage','stage.bmp','unit');self.assertEqual(r['optical_verification']['attempt'],2)
        self.assertEqual(owner.recover.call_count,1)
        g.check=Mock(return_value={'passed':False})
        with self.assertRaises(RuntimeError):g.ensure('stage','stage.bmp','unit')
        self.assertEqual(g.check.call_count,2)
    def test_postcheck_never_rewrites_phase(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);phase=out/'phase.bmp';phase.write_bytes(b'phase')
            raw=self.a.astype(np.uint8);meta={'camera':{k:{'value':v} for k,v in {'ExposureTime':'150','Gain':'Gain_X4','AcquisitionFrameRate':'100','PixelFormat':'Mono8'}.items()}}
            v,s=fingerprint(raw,[0,0,96,96]);probe=Mock();probe.capture.return_value=(raw,meta,out/'raw.png')
            bank={'references':{'stage':v.tolist(),'lens':self.b.tolist()},'roi_xyxy':[0,0,96,96],
                  'camera_actual':{k:x['value'] for k,x in meta['camera'].items()}}
            owner=Mock();g=Guard(owner,probe,bank,out,{})
            self.assertTrue(g.postcheck('stage',phase,'post')['passed']);owner.show.assert_not_called()
    def test_approved_bank_pins_everything(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);lut=out/'x.lut';lut.write_bytes(b'lut');phase=out/'p.bmp';phase.write_bytes(b'phase')
            c={'camera':'config'};b={'approved':False,'hardware_config_sha256':digest(c),'lut_sha256':sha(lut),
                                    'phase_paths':{'stage':str(phase)},'phase_sha256':{'stage':sha(phase)}}
            b['bank_id']=digest(b);bank=out/'bank.json';write(bank,b)
            with self.assertRaises(ValueError):load_bank(bank,c,{'phase_lut':str(lut)})
            b['approved']=True;b['approval']={'bank_id':b['bank_id']};write(bank,b)
            self.assertEqual(load_bank(bank,c,{'phase_lut':str(lut)})['bank_id'],b['bank_id'])
            b['hardware_config_sha256']='edited';write(bank,b)
            with self.assertRaises(ValueError):load_bank(bank,c,{'phase_lut':str(lut)})

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.session=self.root/'sessions/s';self.folder=self.session/'ccd/image_a';self.folder.mkdir(parents=True)
        self.mf={'phase_sha256':'phase','hardware_identity':'hw','entries':[{'id':'image_a'}]}
        write(self.session/'play/vision_router/manifest.json',self.mf)
        self.spec={'session':'s','batch_id':'a'*32,'stage':'vision_router','ids':['image_a'],
          'phase_receipt':{'phase_sha256':'phase','optical_verification':{'passed':True,'target':'vision_router','bank_id':'bank'}}}
    def tearDown(self):self.temp.cleanup()
    def test_unverified_and_existing_rejected(self):
        self.spec['phase_receipt']['optical_verification']['passed']=False
        with self.assertRaises(ValueError):initialize(self.spec,self.root)
        self.spec['phase_receipt']['optical_verification']['passed']=True
        (self.folder/'vision_router.png').write_bytes(b'existing')
        with self.assertRaises(ValueError):initialize(self.spec,self.root)
    def test_quarantine_preserves_other_stages_and_idempotent(self):
        initialize(self.spec,self.root)
        with self.assertRaises(ValueError):assert_no_pending(self.session)
        for suffix in ('.png','.record.json','.json'):(self.folder/('vision_router'+suffix)).write_bytes(b'new')
        (self.folder/'vision_global.png').write_bytes(b'old')
        quarantine(self.spec,self.root);quarantine(self.spec,self.root)
        self.assertTrue((self.folder/'vision_global.png').exists())
        self.assertFalse((self.folder/'vision_router.png').exists())
        self.assertEqual((self.session/'quarantine'/('a'*32)/'image_a/vision_router.png').read_bytes(),b'new')
        assert_no_pending(self.session)
    def test_path_escape(self):
        self.spec['ids']=['../escape']
        with self.assertRaises(ValueError):initialize(self.spec,self.root)
    def test_accept_requires_integrity_and_postcheck(self):
        initialize(self.spec,self.root)
        path=self.folder/'vision_router.png';path.write_bytes(b'new')
        write(self.folder/'vision_router.record.json',{'phase_sha256':'phase','hardware_identity':'hw','files':{'vision_router.png':sha(path)}})
        self.spec['verification']={'bank_id':'bank','postcheck':{'passed':False,'target':'vision_router'}}
        with self.assertRaises(ValueError):accept(self.spec,self.root)
        self.spec['verification']['postcheck']['passed']=True;accept(self.spec,self.root)
        with self.assertRaises(ValueError):quarantine(self.spec,self.root)
        assert_no_pending(self.session)

if __name__=='__main__':unittest.main()
